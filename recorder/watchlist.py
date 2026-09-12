"""Кого писать прямо сейчас: кандидаты бота, его портфель и хвост после выхода.

Состав корзины моментум-бота меняется каждый цикл, и фиксированный список пар
отвечает не на тот вопрос. Боту нужна книга вокруг его собственных сделок, а
главное — ДО входа: единственное, ради чего запись ему сдалась, это проверить,
не врёт ли мгновенный стакан, по которому он решает входить в пару или
пропустить её. Подписаться в момент заявки поздно — пока придёт снимок книги,
сделка уже сделана.

Поэтому набор складывается из трёх частей:

  кандидаты — верх и низ рейтинга, посчитанного здесь по тем же правилам, что
              у бота: доходность за WATCH_LOOKBACK_HOURS среди пар выше порога
              оборота. Из них он корзину и берёт, причём не только по
              расписанию: когда корзина пустеет, пересборка идёт внепланово, и
              предсказать её нельзя. Поэтому кандидаты держатся всегда, а не
              за полчаса до сетки;
  портфель  — что бот держит сейчас, из его файла состояния;
  хвост     — выпавшая пара остаётся ещё WATCH_TAIL_MINUTES: после выхода по
              стопу начинается самое интересное, отскок или продолжение.

Зависимость от чужого проекта здесь сознательная и односторонняя: читается
файл, бот о записи не знает. Если файл не прочитался или биржа не ответила,
возвращается прошлый набор — молча писать не то хуже, чем писать прежнее.
"""

import json
import logging
import time

from config import (
    BOT_STATE_PATH, WATCH_CANDIDATES, WATCH_LOOKBACK_HOURS,
    WATCH_MIN_TURNOVER, WATCH_TAIL_MINUTES,
)
from recorder.costmap import momentum, universe

log = logging.getLogger("recorder")


def to_recorder(symbol):
    """MEXC отдаёт боту `NIULAI/USDT:USDT`, а записи нужен `NIULAI_USDT`."""
    base = symbol.split("/")[0].strip()
    quote = symbol.split("/")[1].split(":")[0] if "/" in symbol else "USDT"
    return f"{base}_{quote}"


def portfolio():
    """Что бот держит прямо сейчас. None, если состояние не прочиталось."""
    try:
        state = json.loads(BOT_STATE_PATH.read_text())
    except FileNotFoundError:
        log.warning("состояние бота не найдено: %s", BOT_STATE_PATH)
        return None
    except Exception as exc:
        log.warning("состояние бота не прочиталось: %s", exc)
        return None
    positions = state.get("positions")
    if not isinstance(positions, dict):
        log.warning("в состоянии бота нет словаря positions — формат изменился?")
        return None
    return {to_recorder(s) for s in positions}


def candidates(pause=0.2):
    """Верх и низ рейтинга — то, из чего бот соберёт корзину.

    Считается по публичным данным теми же правилами, что у бота. Обход стоит
    полторы сотни запросов, поэтому вызывается редко и в отдельном потоке.
    """
    pool = [s for s, _, _ in universe(min_turnover=WATCH_MIN_TURNOVER)]
    if not pool:
        return None
    scored = []
    for symbol in pool:
        try:
            value = momentum(symbol, WATCH_LOOKBACK_HOURS)
            if value is not None:
                scored.append((symbol, value))
        except Exception as exc:
            log.debug("%s: рейтинг не посчитался — %s", symbol, exc)
        time.sleep(pause)
    if len(scored) < WATCH_CANDIDATES * 4:
        log.warning("рейтинг посчитан только по %d парам — оставляю прежний набор",
                    len(scored))
        return None
    scored.sort(key=lambda r: -r[1])
    top = {s for s, _ in scored[:WATCH_CANDIDATES]}
    bottom = {s for s, _ in scored[-WATCH_CANDIDATES:]}
    log.info("рейтинг по %d парам: верх %s | низ %s", len(scored),
             ", ".join(sorted(top)), ", ".join(sorted(bottom)))
    return top | bottom


class Watchlist:
    """Текущий набор символов и то, как он менялся."""

    def __init__(self, fallback):
        self.fallback = list(fallback)      # если бот недоступен — пишем это
        self.candidates = set()
        self.held = set()
        self.leaving = {}                   # символ -> когда снимать подписку
        self.current = set(fallback)

    def refresh(self):
        """Пересобрать набор. Возвращает (добавить, снять)."""
        fresh = candidates()
        if fresh is not None:
            self.candidates = fresh

        held = portfolio()
        if held is not None:
            gone = self.held - held
            deadline = time.time() + WATCH_TAIL_MINUTES * 60
            for symbol in gone:
                self.leaving.setdefault(symbol, deadline)
            for symbol in held:
                self.leaving.pop(symbol, None)
            self.held = held

        now = time.time()
        self.leaving = {s: t for s, t in self.leaving.items() if t > now}

        wanted = self.candidates | self.held | set(self.leaving)
        if not wanted:
            # Ни рейтинга, ни состояния: не отключаем запись совсем.
            wanted = set(self.fallback)

        add = wanted - self.current
        drop = self.current - wanted
        self.current = wanted
        return add, drop

    def describe(self):
        return (f"{len(self.current)} пар: портфель {len(self.held)}, "
                f"кандидатов {len(self.candidates)}, хвост {len(self.leaving)}")
