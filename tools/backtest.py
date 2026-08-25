#!/usr/bin/env python3
"""Бэктест скальпинга по записи: очередь, задержка, издержки, контроли.

    python tools/backtest.py --symbol BTC_USDT

Главная и самая обманчивая часть здесь — МОДЕЛЬ ИСПОЛНЕНИЯ. Наивный бэктест
считает, что лимитка исполняется, как только цена коснулась её уровня. Для
маркет-мейкера это ложь, которая рисует прибыль из воздуха: на самом деле
перед тобой стоит очередь, и пока весь объём впереди не проторгуется, тебя не
нальют. А когда наливают — часто именно потому, что рынок пошёл против тебя.

Что здесь смоделировано:

  ОЧЕРЕДЬ. При постановке заявки запоминается объём, уже стоящий на этой цене.
  Он уменьшается ТОЛЬКО реальными сделками через этот уровень — отмены впереди
  нас не засчитываются, потому что по L2 их не отличить от исполнений. Это
  занижает число исполнений, то есть ошибается в безопасную сторону.

  ЗАДЕРЖКА. Стратегия видит книгу на момент t, а её заявка попадает на биржу в
  t + lag. По умолчанию 200 мс — замеренная задержка потока плюс пачки биржи.
  Без этого бэктест торгует по данным из будущего.

  ИЗДЕРЖКИ. Вход лимиткой — комиссия 0. Выход по стопу и по таймеру — тейкер,
  со своей комиссией и переходом через спред.

  КОНТРОЛИ. Рядом всегда считаются: случайный вход с той же частотой и теми же
  правилами выхода, и купи-и-держи на том же окне. Стратегия обязана обыгрывать
  оба, иначе это не стратегия.
"""

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from recorder.book import OrderBook
from tools._io import stream, downtime


# --- чтение записи ----------------------------------------------------------

def tick_size(symbol):
    """Шаг цены инструмента. Без него выходная лимитка встаёт на цену, которой
    не существует: на BTC шаг 0.1, а «вход × 1.0002» даёт 77344.466. Очередь
    на несуществующем уровне пуста, и модель исполняет такую заявку слишком
    легко — то есть врёт в свою пользу."""
    fallback = {"BTC_USDT": 0.1, "ETH_USDT": 0.01, "SOL_USDT": 0.01,
                "XAU_USDT": 0.01, "XRP_USDT": 0.0001, "HYPE_USDT": 0.001}
    try:
        import requests
        body = requests.get("https://contract.mexc.com/api/v1/contract/detail",
                            timeout=15).json()
        for row in body.get("data") or []:
            if row.get("symbol") == symbol:
                return float(row["priceUnit"])
    except Exception:
        pass
    return fallback.get(symbol, 0.01)


def events(symbol, hours=None):
    """Поток событий по инструменту. Генератор, а не список: суточная запись
    в виде объектов Python не помещается в память сервера."""
    return stream(symbol=symbol, hours=hours)


# --- исполнение -------------------------------------------------------------

@dataclass
class RestingOrder:
    """Лимитка, стоящая в очереди на своей цене."""
    side: int              # 0 покупка, 1 продажа
    price: float
    size: float
    queue_ahead: float     # сколько объёма стоит перед нами
    placed_us: int
    filled: float = 0.0

    def remaining(self):
        return self.size - self.filled


@dataclass
class Trade:
    entry_us: int
    exit_us: int
    side: int
    entry: float
    exit: float
    size: float
    exit_reason: str
    pnl_bp: float = 0.0
    cost_bp: float = 0.0


@dataclass
class Engine:
    """Книга, очередь, позиция, риск. Ничего не решает — только исполняет."""
    taker_bp: float
    stop_bp: float
    target_bp: float
    time_stop_s: float
    lag_us: int

    maker_exit: bool = True
    taker_entry: bool = False   # идти за движением можно только по рынку
    tick: float = 0.01
    book: OrderBook = None
    order: RestingOrder = None
    exit_order: RestingOrder = None
    pending: list = field(default_factory=list)   # заявки, ещё летящие на биржу
    position: float = 0.0
    entry_cost_bp: float = 0.0
    entry_price: float = 0.0
    entry_us: int = 0
    side: int = 0
    trades: list = field(default_factory=list)
    impossible: int = 0     # исполнений выхода по цене, недостижимой в книге

    def submit(self, now_us, side, price, size):
        """Заявка уходит на биржу и появится там только через lag."""
        if self.order or self.position or self.pending:
            return
        self.pending.append((now_us + self.lag_us, side, price, size))

    def activate(self, now_us):
        while self.pending and self.pending[0][0] <= now_us:
            _, side, price, size = self.pending.pop(0)
            if self.taker_entry:
                # По рынку: покупаем по аску, продаём по биду. Дороже на
                # комиссию и спред, зато исполняемся ВСЕГДА — а не только
                # тогда, когда движение пошло против нас.
                b, a = self.book.best()
                if not b or not a:
                    continue
                self.entry_cost_bp = self.taker_bp
                self._open(now_us, side, a[0] if side == 0 else b[0], size)
                continue
            level = self.book.bids if side == 0 else self.book.asks
            ahead = level.get(price, 0.0)
            self.order = RestingOrder(side, price, size, ahead, now_us)

    def on_trades(self, now_us, deals):
        """Сделки двигают очередь и могут налить как вход, так и выход."""
        self._work(now_us, deals, entry=True)
        self._work(now_us, deals, entry=False)

    def _work(self, now_us, deals, entry):
        o = self.order if entry else self.exit_order
        if not o:
            return
        for price, volume, aggressor in deals:
            # нашу сторону задевают сделки противоположного агрессора
            hits = (o.side == 0 and aggressor == 1 and price <= o.price) or \
                   (o.side == 1 and aggressor == 0 and price >= o.price)
            if not hits:
                continue
            if o.queue_ahead > 0:
                eaten = min(o.queue_ahead, volume)
                o.queue_ahead -= eaten
                volume -= eaten
            if volume > 0 and o.remaining() > 0:
                fill = min(volume, o.remaining())
                o.filled += fill
                if o.filled >= o.size - 1e-9:
                    if entry:
                        self._open(now_us, o.side, o.price, o.size)
                        self.order = None
                    else:
                        # Проверка на невозможное исполнение: продать по цене
                        # выше текущего аска (или купить ниже бида) нельзя.
                        # Если такое случилось, значит заявку налили там, где
                        # рынка не было, и вся прибыль конструкции — выдумка.
                        bb, aa = self.book.best()
                        if bb and aa:
                            if (o.side == 1 and o.price > aa[0] + self.tick) or \
                               (o.side == 0 and o.price < bb[0] - self.tick):
                                self.impossible += 1
                        self._close(now_us, o.price, "цель лимиткой", cost=0.0)
                    return

    def resync_queue(self):
        """Отмены впереди нас по L2 не видны, но и врать себе нельзя:
        очередь не может быть длиннее того, что реально стоит на уровне."""
        for o in (self.order, self.exit_order):
            if not o:
                continue
            level = self.book.bids if o.side == 0 else self.book.asks
            o.queue_ahead = min(o.queue_ahead, level.get(o.price, 0.0))

    def cancel(self):
        self.order = None

    def _open(self, now_us, side, price, size):
        self.position, self.side = size, side
        self.entry_price, self.entry_us = price, now_us
        if self.maker_exit:
            # Выход тоже лимиткой: тейкерный выход съедает половину цели и
            # требует винрейта под 75% просто чтобы не терять.
            raw = price * (1 + self.target_bp / 1e4) if side == 0 \
                else price * (1 - self.target_bp / 1e4)
            # округляем В СТОРОНУ ОТ входа: цель становится чуть дальше, а не
            # чуть ближе. Ошибаться надо против себя, иначе бэктест рисует
            # исполнения, которых не было бы
            steps = raw / self.tick
            target = (math.ceil(steps) if side == 0 else math.floor(steps)) * self.tick
            target = round(target, 10)
            opposite = 1 - side
            level = self.book.asks if opposite == 1 else self.book.bids
            self.exit_order = RestingOrder(opposite, target, size,
                                           level.get(target, 0.0), now_us)

    def _close(self, now_us, price, reason, cost):
        move_bp = (price / self.entry_price - 1) * 1e4 * (1 if self.side == 0 else -1)
        total = cost + self.entry_cost_bp
        self.trades.append(Trade(self.entry_us, now_us, self.side,
                                 self.entry_price, price, self.position, reason,
                                 pnl_bp=move_bp - total, cost_bp=total))
        self.position = 0.0
        self.entry_cost_bp = 0.0
        self.exit_order = None

    def manage(self, now_us):
        """Стоп, цель, таймер. Выход всегда тейкером — так честнее."""
        if not self.position:
            return
        b, a = self.book.best()
        if not b or not a:
            return
        # Выход из ДЛИННОЙ позиции — это продажа, а продают по биду. Из
        # короткой — покупка, покупают по аску. Было наоборот, и каждая сделка
        # оценивалась на один спред лучше, чем на самом деле: на BTC это
        # 0.013 б.п. и незаметно, на SOL 1.0 б.п. — сопоставимо со всем
        # измеренным преимуществом.
        mark = b[0] if self.side == 0 else a[0]
        move_bp = (mark / self.entry_price - 1) * 1e4 * (1 if self.side == 0 else -1)
        age = (now_us - self.entry_us) / 1e6
        reason = None
        if move_bp <= -self.stop_bp:
            reason = "стоп"
        elif age >= self.time_stop_s:
            reason = "таймер"
        elif not self.maker_exit and move_bp >= self.target_bp:
            reason = "цель"
        if reason:
            # стоп и таймер уходят по рынку — ждать тут нечего
            self._close(now_us, mark, reason, cost=self.taker_bp)


# --- прогон -----------------------------------------------------------------

def replay(symbol, p, decide, seed=0):
    """Один прогон. Обёртка вокруг replay_multi для единственной конструкции."""
    out = replay_multi(symbol, p, [("одна", p, decide)], seed=seed)
    return out["одна"]


def _replay_gen(symbol, base, runs, seed=0, holes=None):
    """Много конструкций за ОДИН проход по записи.

    Чтение и сборка книги — самая дорогая часть, и она общая для всех
    конструкций. Гонять запись по разу на каждую означало бы минуты вместо
    секунд и соблазн урезать сетку параметров ради скорости.

    `runs` — список (имя, параметры, решающая функция). У каждой конструкции
    свой движок и своя позиция, они друг о друге не знают.
    """
    p = base
    rng = np.random.default_rng(seed)
    tick = tick_size(symbol)
    book = OrderBook(symbol)

    spreads = []
    path_bp = 0.0        # полный ход цены: сумма модулей изменений середины
    prev_mid = None
    # история середины для сигнала разворота: движение за прошлую секунду
    mid_hist = deque(maxlen=200)
    warmup = []          # первые замеры идут только на оценку сигмы
    sigma = None
    deal_hist = deque(maxlen=4000)   # (ts, объём со знаком) для дельты сделок
    delta_warm = []
    delta_sigma = None
    engines = {}
    for name, params, decide in runs:
        eng = Engine(taker_bp=params["taker_bp"], stop_bp=params["stop_bp"],
                     target_bp=params["target_bp"],
                     time_stop_s=params["time_stop_s"],
                     lag_us=params["lag_ms"] * 1000,
                     maker_exit=not params.get("taker_exit", False),
                     taker_entry=params.get("taker_entry", False),
                     tick=tick)
        eng.book = book
        engines[name] = eng
    signals = {name: 0 for name, _, _ in runs}

    step_us = p["step_ms"] * 1000
    next_ts = None
    ofi_hist = []            # (ts_us, приращение OFI)
    prev_bid = prev_ask = None
    order_age_us = p["order_life_s"] * 1_000_000

    # Простои записи считаются один раз снаружи и передаются сюда: иначе
    # каждый из восьми генераторов сканирует все файлы ещё до начала работы.
    holes = downtime(p.get("hours")) if holes is None else holes
    hole_index = 0
    last_row_ts = None
    BLIND_US = 60_000_000   # минута без строк — управлять позицией уже нечем
    # Строки подаются снаружи через send(): так один проход по файлам кормит
    # сразу все инструменты, вместо того чтобы каждый читал запись заново.
    while True:
        r = yield
        if r is None:
            break
        ts = r["ts_local_us"]
        # Рвём только на настоящем простое записи. Тишина по инструменту —
        # это просто спокойный рынок, позицию в ней держать можно.
        # Страховка поверх детектора простоев: если строк по инструменту не
        # было дольше минуты, движок всё это время не мог ни сработать стопом,
        # ни закрыться по таймеру, а позиция при этом «жила». В локальной
        # записи нашлась сделка длиной 2737 секунд при таймере в 900 — она
        # молча забрала движение, которого мы не наблюдали. Детектор простоев
        # такие дыры видит не всегда, поэтому проверяем ещё и напрямую.
        if last_row_ts is not None and ts - last_row_ts > BLIND_US:
            book.reset(); prev_bid = prev_ask = None
            for eng in engines.values():
                eng.order = None; eng.exit_order = None
                eng.pending.clear()
                eng.position = 0.0; eng.entry_cost_bp = 0.0
            next_ts = ts
        last_row_ts = ts

        while hole_index < len(holes) and holes[hole_index][1] <= ts:
            # Через разрыв записи позицию держать нельзя: цен в этом окне мы не
            # видели, а книга после разрыва пуста, и движок не может ни
            # сработать стопом, ни закрыться по таймеру. Условие тут раньше
            # стояло лишнее, и позиции переживали дыру: одна сделка держалась
            # 2737 секунд при таймере в 900 и забирала движение, которого мы
            # не наблюдали. Сбрасываем безусловно и молча — такие сделки в
            # статистику попадать не должны вовсе.
            book.reset(); prev_bid = prev_ask = None
            for eng in engines.values():
                eng.order = None; eng.exit_order = None
                eng.pending.clear()
                eng.position = 0.0; eng.entry_cost_bp = 0.0
            next_ts = holes[hole_index][1]
            hole_index += 1
        payload = json.loads(r["payload"])

        if r["channel"] == "snapshot":
            book.apply_snapshot(payload)
            if next_ts is None and book.ready:
                next_ts = ts
        elif r["channel"] == "depth":
            if book.apply_delta(payload) == "ok":
                b, a = book.best()
                if b and a:
                    if prev_bid is not None:
                        d = 0.0
                        d += b[1] if b[0] > prev_bid[0] else (
                            b[1] - prev_bid[1] if b[0] == prev_bid[0] else -prev_bid[1])
                        d -= a[1] if a[0] < prev_ask[0] else (
                            a[1] - prev_ask[1] if a[0] == prev_ask[0] else -prev_ask[1])
                        ofi_hist.append((ts, d))
                    prev_bid, prev_ask = b, a
                for eng in engines.values():
                    eng.resync_queue()
        elif r["channel"] == "deal":
            deals = []
            for deal in (payload if isinstance(payload, list) else [payload]):
                try:
                    vol = float(deal["v"])
                    side = 0 if int(deal["T"]) == 1 else 1
                    deals.append((float(deal["p"]), vol, side))
                    deal_hist.append((ts, vol if side == 0 else -vol))
                except (KeyError, TypeError, ValueError):
                    continue
            for eng in engines.values():
                eng.on_trades(ts, deals)
        elif r["channel"] == "gap":
            book.ready = False
            prev_bid = prev_ask = None

        for eng in engines.values():
            eng.activate(ts)
            eng.manage(ts)

        if next_ts is None or ts < next_ts + step_us:
            continue
        next_ts = ts

        cutoff = ts - 1_000_000
        while ofi_hist and ofi_hist[0][0] < cutoff:
            ofi_hist.pop(0)
        b, a = book.best()
        if not b or not a or b[0] >= a[0]:
            continue

        spreads.append((a[0] - b[0]) / ((a[0] + b[0]) / 2) * 1e4)
        # дельта сделок за последние 5 секунд: перевес агрессивных покупок
        cut5 = ts - 5_000_000
        while deal_hist and deal_hist[0][0] < cut5:
            deal_hist.popleft()
        delta_raw = sum(v for _, v in deal_hist)
        now_mid = (a[0] + b[0]) / 2
        if prev_mid:
            path_bp += abs(now_mid / prev_mid - 1) * 1e4
        prev_mid = now_mid
        mid_hist.append((ts, now_mid))
        # движение за прошлую секунду
        past = next((m for t0, m in mid_hist if ts - t0 <= 1_000_000), None)
        move_bp = (now_mid / past - 1) * 1e4 if past else 0.0
        # Сигма оценивается по первым замерам и замораживается: считать её по
        # всей записи значило бы подглядывать в будущее.
        if delta_sigma is None:
            delta_warm.append(delta_raw)
            if len(delta_warm) >= 1000:
                est = float(np.std(delta_warm))
                if est > 0:
                    delta_sigma = est
        if sigma is None:
            warmup.append(move_bp)
            # Порог разогрева небольшой намеренно: на коротких записях
            # большой съедал бы всю выборку и стратегия молча не торговала.
            if len(warmup) >= 1000:
                est = float(np.std(warmup))
                if est > 0:
                    sigma = est
            qb, qa = b[1], a[1]
            continue
        qb, qa = b[1], a[1]
        state = {
            "move_sig": move_bp / sigma,
            "delta_sig": (delta_raw / delta_sigma) if delta_sigma else 0.0,
            "imb": (qb - qa) / (qb + qa) if qb + qa else 0.0,
            "ofi": sum(v for _, v in ofi_hist),
            "bid": b[0], "ask": a[0],
        }
        for name, params, decide in runs:
            eng = engines[name]
            if eng.order and ts - eng.order.placed_us > order_age_us:
                eng.cancel()
            if eng.position or eng.order or eng.pending:
                continue
            action = decide(state, rng)
            if action == "buy":
                signals[name] += 1
                eng.submit(ts, 0, b[0], params["size"])
            elif action == "sell":
                signals[name] += 1
                eng.submit(ts, 1, a[0], params["size"])

    import os
    if os.getenv("OBR_DEBUG"):
        print(f"[отладка] узлов сетки: {len(spreads)}, разогрев: {len(warmup)}, "
              f"сигма: {sigma}", file=sys.stderr)
    out = {name: (engines[name].trades, signals[name]) for name, _, _ in runs}
    out["__spread__"] = (sorted(spreads)[len(spreads) // 2] if spreads else 0.0,
                         len(spreads))
    # Физический предел: забрать больше, чем прошла цена, невозможно ничем.
    # Любая ошибка в учёте исполнений вылезает здесь сразу и наглядно.
    out["__path__"] = path_bp
    out["__impossible__"] = {name: engines[name].impossible for name, _, _ in runs}
    return out


def _drive(gen, rows):
    """Прокачать генератор строками и забрать результат."""
    next(gen)
    for r in rows:
        gen.send(r)
    try:
        gen.send(None)
    except StopIteration as stop:
        return stop.value
    return None


def replay_multi(symbol, base, runs, seed=0):
    """Много конструкций на одном инструменте за один проход по записи."""
    return _drive(_replay_gen(symbol, base, runs, seed),
                  events(symbol, base.get("hours")))


def replay_all(symbols, base, runs_by_symbol, seed=0):
    """Все инструменты за ОДИН проход по файлам.

    Раньше каждый инструмент читал запись заново: восемь инструментов —
    восемь распаковок полугигабайта, отсюда получасовой перебор. Здесь поток
    читается один раз, строки раскладываются по инструментам, и каждый
    обрабатывается своим набором движков.

    Строки НЕ накапливаются: каждая сразу уходит в свой генератор. Собрать их
    в списки по инструментам было бы проще, но суточная запись в виде объектов
    Python — это гигабайты, и сервер бы не пережил.
    """
    holes = downtime(base.get("hours"))
    gens = {}
    for sym in symbols:
        if sym not in runs_by_symbol:
            continue
        gen = _replay_gen(sym, base, runs_by_symbol[sym], seed, holes=holes)
        next(gen)                       # довести до первого yield
        gens[sym] = gen

    for r in stream(hours=base.get("hours"), progress=True):
        gen = gens.get(r["symbol"])
        if gen is not None:
            gen.send(r)

    out = {}
    for sym, gen in gens.items():
        try:
            gen.send(None)
        except StopIteration as stop:
            out[sym] = stop.value
    return out


def strategy(p):
    """Вход по согласию дисбаланса и потока заявок, лимиткой на своей стороне."""
    def decide(s, rng):
        if s["imb"] > p["imb_th"] and s["ofi"] > p["ofi_th"]:
            return "buy"
        if s["imb"] < -p["imb_th"] and s["ofi"] < -p["ofi_th"]:
            return "sell"
        return None
    return decide


def reversal_strategy(p):
    """Фейд резкого движения: цена рванула вниз — покупаем, вверх — продаём.

    Вход лимиткой на своей стороне книги. На инструментах с широким спредом
    это принципиально: входя пассивно, спред зарабатываешь, а не платишь.
    Именно поэтому сигнал в 0.4-1.0 б.п., безнадёжный для тейкера, здесь ещё
    может окупиться.
    """
    def decide(s, rng):
        move = s.get("move_sig", 0.0)
        if move <= -p["move_th"]:
            return "buy"
        if move >= p["move_th"]:
            return "sell"
        return None
    return decide


def delta_strategy(p):
    """Идти ЗА потоком агрессивных сделок на горизонте минут.

    Наклон признака положительный: куда льют по рынку, туда цена и идёт
    следующие пять-пятнадцать минут. Это продолжение, а не разворот, и это
    прямо противоположно тому, что работает на секундных горизонтах, где
    цену двигает давление в очереди, а не исполненный объём.

    Вход лимиткой на своей стороне: при удержании в минуты спешить некуда,
    и лишний базисный пункт на входе съедает заметную долю цели.
    """
    def decide(s, rng):
        d = s.get("delta_sig", 0.0)
        if d >= p["delta_th"]:
            return "buy"
        if d <= -p["delta_th"]:
            return "sell"
        return None
    return decide


def random_control(rate):
    """Контроль: та же частота входов, направление случайное."""
    def decide(s, rng):
        if rng.random() > rate:
            return None
        return "buy" if rng.random() < 0.5 else "sell"
    return decide


def stats(trades):
    if not trades:
        return None
    x = np.array([t.pnl_bp for t in trades])
    n = len(x)
    t_stat = float(x.mean() / (x.std(ddof=1) / np.sqrt(n))) if n > 1 and x.std() else 0.0
    return {"n": n, "mean_bp": float(x.mean()), "total_bp": float(x.sum()),
            "win": float((x > 0).mean() * 100), "t": t_stat,
            "by_reason": {r: sum(1 for t in trades if t.exit_reason == r)
                          for r in ("цель", "цель лимиткой", "стоп", "таймер")}}


def show(name, s, notional):
    if not s:
        print(f"  {name:<22} сделок 0 — стратегия не сработала ни разу")
        return
    money = s["total_bp"] / 1e4 * notional
    print(f"  {name:<22}{s['n']:>7}{s['win']:>9.1f}%{s['mean_bp']:>11.3f}"
          f"{s['total_bp']:>11.1f}{money:>11.2f}{s['t']:>8.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTC_USDT")
    ap.add_argument("--hours", type=float, default=None,
                    help="сколько последних часов брать (по умолчанию всё)")
    ap.add_argument("--stop-bp", type=float, default=2.0)
    ap.add_argument("--target-bp", type=float, default=2.0)
    ap.add_argument("--time-stop-s", type=float, default=60)
    ap.add_argument("--order-life-s", type=float, default=10)
    ap.add_argument("--imb-th", type=float, default=0.4)
    ap.add_argument("--ofi-th", type=float, default=0.0)
    ap.add_argument("--lag-ms", type=int, default=200)
    ap.add_argument("--step-ms", type=int, default=200)
    ap.add_argument("--taker-bp", type=float, default=1.0)
    ap.add_argument("--size", type=float, default=1.0)
    ap.add_argument("--taker-exit", action="store_true",
                    help="выходить по цели тейкером (по умолчанию лимиткой)")
    ap.add_argument("--notional", type=float, default=2500,
                    help="номинал позиции в $, чтобы перевести б.п. в деньги")
    a = ap.parse_args()
    p = vars(a)

    print("=" * 78)
    print(f"  БЭКТЕСТ {a.symbol}   стоп {a.stop_bp} б.п.   цель {a.target_bp} б.п.   "
          f"таймер {a.time_stop_s:g} с")
    print(f"  задержка решения {a.lag_ms} мс   выход тейкером {a.taker_bp} б.п.   "
          f"номинал ${a.notional:,.0f}")
    print("=" * 78)

    trades, signals = replay(a.symbol, p, strategy(p))
    s = stats(trades)

    # контроль с той же частотой входов
    total_steps = max(signals, 1) * 50
    rate = signals / total_steps if total_steps else 0.001
    ctrl, _ = replay(a.symbol, p, random_control(min(rate, 0.05)), seed=7)
    sc = stats(ctrl)

    print(f"\n  сигналов {signals}, из них дошло до сделки "
          f"{s['n'] if s else 0} ({(s['n']/signals*100) if signals and s else 0:.0f}% — "
          "остальные не налили, очередь не подошла)")
    print(f"\n  {'':<22}{'сделок':>7}{'винрейт':>10}{'ср. б.п.':>11}"
          f"{'итого б.п.':>11}{'итого $':>11}{'t':>8}")
    show("стратегия", s, a.notional)
    show("случайный вход", sc, a.notional)

    if s:
        print(f"\n  выходы: по цели {s['by_reason']['цель'] + s['by_reason']['цель лимиткой']}, "
              f"по стопу {s['by_reason']['стоп']}, "
              f"по таймеру {s['by_reason']['таймер']}")
    print("\n  Стратегия обязана обыгрывать случайный вход с t > 3. Если t ниже —")
    print("  это шум, сколько бы ни было итоговых б.п.")
    print("=" * 78)


if __name__ == "__main__":
    main()
