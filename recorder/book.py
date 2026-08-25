"""Сборка книги заявок из инкрементов + обнаружение потерь.

Как это устроено у MEXC (проверено на живом потоке 2026-08-23):

  * каждое сообщение стакана несёт `begin` и `end` — диапазон внутренних
    номеров изменений, склеенных в одну пачку. `version` == `end`;
  * непрерывность потока означает `begin == предыдущий end + 1`.
    Шаг самого `version` при этом произвольный (от 3 до нескольких сотен) —
    сравнивать надо именно begin с прошлым end, иначе получишь сплошные
    «разрывы» на совершенно целом потоке;
  * пачки приходят примерно раз в 200 мс, то есть биржа сама агрегирует
    изменения за это окно.

Стыковка со снимком — место, где всё обычно и ломается. REST-снимок ОТСТАЁТ
от живого потока: пока он едет по сети, биржа успевает отправить ещё пачки,
и его версия V оказывается ниже текущей позиции стрима. Применить его «как
есть» и ждать продолжения нельзя — следующая живая пачка придёт с begin
намного больше V+1, и это будет выглядеть как вечный разрыв.

Поэтому:
  * с момента запроса снимка складываем приходящие пачки в буфер;
  * получив снимок на версии V, выбрасываем из буфера всё с end <= V;
  * доигрываем начиная с пачки, которая накрывает V+1 (begin <= V+1 <= end);
  * если самая ранняя уцелевшая пачка идёт с begin > V+1 — снимок оказался
    старше буфера, берём новый, буфер не выбрасываем.

Живая книга нужна ТОЛЬКО чтобы заметить разрыв и вовремя переснять. Выводы
делаются офлайн по сырым записям, поэтому ошибка здесь не портит собранные
данные — она портит максимум частоту переснимков.
"""

from dataclasses import dataclass, field


OK = "ok"                    # применено, книга целая
STALE = "stale"              # пачка уже внутри снимка, пропускаем
GAP = "gap"                  # поток порвался: были готовы, номера не сошлись
NOSYNC = "nosync"            # снимка ещё нет, копим в буфер — это не разрыв
UNKNOWN = "unknown"          # нет номеров, проверить целостность нечем
SNAPSHOT_OLD = "snapshot_old"  # снимок отстал от буфера, нужен свежее

# Потолок буфера пачек на время загрузки снимка. При пачке раз в 200 мс это
# больше двух минут ожидания — если не уложились, проблема не в буфере.
MAX_BUFFER = 600


def extract_version(data):
    """Номер последнего изменения в пачке (`end`, он же `version`)."""
    if not isinstance(data, dict):
        return None
    for key in ("version", "end", "v"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _levels(data, side):
    for level in data.get(side) or []:
        if len(level) >= 2:
            yield float(level[0]), float(level[1])


@dataclass
class OrderBook:
    symbol: str
    bids: dict = field(default_factory=dict)   # цена -> объём
    asks: dict = field(default_factory=dict)
    version: int = -1        # номер последнего применённого изменения
    ready: bool = False

    # счётчики для статистики
    _best: tuple = None      # кэш лучших цен, сбрасывается при изменении книги
    applied: int = 0
    gaps: int = 0
    stale: int = 0
    waiting: int = 0
    unversioned: int = 0
    old_snapshots: int = 0
    last_gap: tuple = ()     # (had, begin, end) последнего разрыва
    buffer: list = field(default_factory=list)   # пачки, пришедшие до снимка

    def reset(self):
        """Сброс перед пересинхронизацией. Буфер очищаем: старые пачки
        относятся к книге, которой у нас больше нет."""
        self.bids.clear()
        self.asks.clear()
        self.version = -1
        self.ready = False
        self.buffer.clear()

    def apply_snapshot(self, data):
        """Снимок + доигровка буфера. OK — синхронизировались, SNAPSHOT_OLD —
        снимок старше того, что уже накопили, нужен свежий."""
        version = extract_version(data)
        if version is None:
            return GAP

        buffered, self.buffer = self.buffer, []
        pending = [d for d in buffered if (extract_version(d) or -1) > version]

        if pending:
            first = pending[0]
            begin = first.get("begin")
            if isinstance(begin, (int, float)) and int(begin) > version + 1:
                # Между снимком и буфером дыра: снимок ехал слишком долго.
                # Буфер сохраняем — со следующим снимком он подойдёт.
                self.buffer = pending[-MAX_BUFFER:]
                self.old_snapshots += 1
                return SNAPSHOT_OLD

        self.bids = dict(_levels(data, "bids"))
        self.asks = dict(_levels(data, "asks"))
        self.version = version
        self.ready = True

        for delta in pending:
            self.apply_delta(delta)
        return OK

    def apply_delta(self, data):
        """Инкремент. Возвращает OK / STALE / GAP / UNKNOWN."""
        end = extract_version(data)
        begin = data.get("begin")
        begin = int(begin) if isinstance(begin, (int, float)) else None

        if end is None:
            # Номеров нет — применяем, но участок считается непроверенным.
            self._merge(data)
            self.unversioned += 1
            return UNKNOWN

        if not self.ready:
            # Снимок ещё не пришёл. Копим — без этого буфера снимок никогда
            # не состыкуется с потоком, потому что он приходит с опозданием.
            self.buffer.append(data)
            del self.buffer[:-MAX_BUFFER]
            self.waiting += 1
            return NOSYNC

        if end <= self.version:
            self.stale += 1
            return STALE

        if begin is not None and begin > self.version + 1:
            # Между тем, что у нас, и тем, что пришло, потерялись изменения.
            # Дальше применять нельзя — книга молча разъедется с биржей.
            self.gaps += 1
            self.last_gap = (self.version, begin, end)
            self.ready = False
            return GAP

        # begin <= version + 1 <= end: нормальное продолжение либо стык со
        # снимком. Повторное применение уровней безвредно — в пачке лежат
        # абсолютные значения объёма, а не приращения.
        self._merge(data)
        self.version = end
        self.applied += 1
        return OK

    def _merge(self, data):
        self._best = None
        for price, volume in _levels(data, "bids"):
            if volume == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = volume
        for price, volume in _levels(data, "asks"):
            if volume == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = volume

    # --- то, что пригодится анализатору ------------------------------------

    def best(self):
        """Лучшие бид и аск: (цена, объём) или None, если сторона пуста.

        Результат кэшируется до следующего изменения книги. Без кэша это
        max() и min() по полутора тысячам уровней на каждый вызов, а
        спрашивают лучшие цены много раз за одну строку — при бэктесте
        семнадцать движков подряд. Так один пересчёт вместо семнадцати.
        """
        if self._best is not None:
            return self._best
        bid = max(self.bids) if self.bids else None
        ask = min(self.asks) if self.asks else None
        self._best = (
            (bid, self.bids[bid]) if bid is not None else None,
            (ask, self.asks[ask]) if ask is not None else None,
        )
        return self._best

    def sane(self):
        """Бид не должен быть выше аска. Если стал — книга битая."""
        bid, ask = self.best()
        if bid is None or ask is None:
            return False
        return bid[0] < ask[0]
