#!/usr/bin/env python3
"""Отработка событий стакана: поглощение, свип с возвратом, осушение.

    python tools/setups.py --symbol BTC_USDT --hours 24

Зачем отдельно от analyze и backtest. Те меряют НЕПРЕРЫВНЫЕ признаки —
дисбаланс, поток заявок — и дают среднее преимущество по всем моментам рынка.
Оно вышло около 0.35 б.п. и издержек не перекрывает.

Но скальпер так не торгует. Он ждёт события и делает 5-20 сделок в сутки, а
не 829. Смысл ожидания в том, что УСЛОВНОЕ преимущество на редком событии
может быть в разы больше среднего. Здесь проверяется именно это.

События:

  ПОГЛОЩЕНИЕ. В уровень бьют агрессивными сделками, суммарно больше чем на
  K объёмов этого уровня, а он стоит и цена не уходит. Значит кто-то крупный
  скупает всё, что в него льют. Ожидание: движение в сторону поглощающего.

  СВИП С ВОЗВРАТОМ. Резкий проход цены через несколько уровней меньше чем за
  секунду, затем возврат обратно. Ожидание: продолжение возврата — топливо
  за уровнем выбито.

  ОСУШЕНИЕ. Крупный уровень исчезает БЕЗ соответствующей торговли: не съели,
  а сняли. Ожидание: движение в сторону образовавшейся пустоты.

Для каждого события меряется будущая доходность в сторону ожидания, с той же
защитой, что и в analyze: отсчёт от момента события плюс задержка решения.
Рядом печатается безусловное среднее — событие обязано его обыгрывать, иначе
это не сетап, а просто момент времени.
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from recorder.book import OrderBook
from tools._io import stream, downtime


def collect(symbol, hours, cfg):
    book = OrderBook(symbol)
    holes = downtime(hours)
    hole_i = 0

    mids = []          # (ts_us, mid) — для расчёта будущей доходности
    events = {"поглощение": [], "свип с возвратом": [], "осушение": []}

    # состояние детекторов
    level_state = {}   # цена -> {"traded": объём, "shown": объём, "since": ts}
    tape = deque(maxlen=4000)
    recent_mid = deque(maxlen=600)     # (ts, mid) последние секунды
    swept = None                       # (ts, направление, цена до свипа)

    for r in stream(symbol=symbol, hours=hours, progress=True):
        ts = r["ts_local_us"]
        while hole_i < len(holes) and holes[hole_i][1] <= ts:
            book.reset(); level_state.clear(); tape.clear()
            recent_mid.clear(); swept = None
            hole_i += 1

        payload = json.loads(r["payload"])
        if r["channel"] == "snapshot":
            book.apply_snapshot(payload)
            continue
        if r["channel"] == "gap":
            book.ready = False
            continue
        if r["channel"] == "deal":
            for d in (payload if isinstance(payload, list) else [payload]):
                try:
                    price, vol = float(d["p"]), float(d["v"])
                    side = 0 if int(d["T"]) == 1 else 1
                except (KeyError, TypeError, ValueError):
                    continue
                tape.append((ts, price, vol, side))
                st = level_state.setdefault(price, {"traded": 0.0, "shown": 0.0,
                                                    "since": ts})
                st["traded"] += vol
            continue
        if r["channel"] != "depth" or book.apply_delta(payload) != "ok":
            continue

        b, a = book.best()
        if not b or not a or b[0] >= a[0]:
            continue
        mid = (b[0] + a[0]) / 2
        mids.append((ts, mid))
        recent_mid.append((ts, mid))

        # запоминаем показанный объём на лучших ценах
        for price, shown in ((b[0], b[1]), (a[0], a[1])):
            st = level_state.setdefault(price, {"traded": 0.0, "shown": shown,
                                                "since": ts})
            st["shown"] = max(st["shown"], shown)

        # --- ПОГЛОЩЕНИЕ ----------------------------------------------------
        for price, shown, direction in ((b[0], b[1], +1), (a[0], a[1], -1)):
            st = level_state.get(price)
            if not st or st["shown"] <= 0:
                continue
            if st["traded"] < cfg["absorb_k"] * st["shown"]:
                continue
            if ts - st["since"] > cfg["absorb_window_s"] * 1_000_000:
                st["traded"] = 0.0; st["since"] = ts
                continue
            # уровень всё ещё лучший и не пробит — значит его держат
            events["поглощение"].append((ts, direction))
            st["traded"] = 0.0
            st["since"] = ts

        # --- СВИП С ВОЗВРАТОМ ----------------------------------------------
        window = [m for t, m in recent_mid if ts - t <= 1_000_000]
        if len(window) > 3:
            lo, hi = min(window), max(window)
            span_bp = (hi - lo) / mid * 1e4
            if swept is None and span_bp >= cfg["sweep_bp"]:
                direction = -1 if mid >= (hi + lo) / 2 else +1   # ждём возврата
                swept = (ts, direction, (hi + lo) / 2)
            elif swept is not None:
                age = (ts - swept[0]) / 1e6
                back = abs(mid - swept[2]) / mid * 1e4
                if age <= cfg["reclaim_s"] and back <= cfg["sweep_bp"] * 0.4:
                    events["свип с возвратом"].append((ts, swept[1]))
                    swept = None
                elif age > cfg["reclaim_s"]:
                    swept = None

        # --- ОСУШЕНИЕ ------------------------------------------------------
        levels = (list(payload.get("bids") or [])[:12]
                  + list(payload.get("asks") or [])[:12])
        for level in levels:
            # уровень приходит как [цена, объём, число заявок]
            price, volume = float(level[0]), float(level[1])
            if volume != 0:
                continue
            st = level_state.get(price)
            if not st or st["shown"] < cfg["wall_size"]:
                continue
            if st["traded"] > st["shown"] * 0.3:
                continue            # съели, а не сняли — это не осушение
            if abs(price - mid) / mid * 1e4 > cfg["wall_near_bp"]:
                continue
            direction = -1 if price < mid else +1   # пустота там, где сняли
            events["осушение"].append((ts, direction))
            level_state.pop(price, None)

        # чистим состояние далёких уровней
        if len(level_state) > 4000:
            level_state = {p: s for p, s in level_state.items()
                           if abs(p - mid) / mid * 1e4 < 50}

    return mids, events


def forward(mids, events, horizons_s, lag_ms):
    ts = np.array([t for t, _ in mids], dtype=np.int64)
    mid = np.array([m for _, m in mids])
    out = {}
    for name, evs in events.items():
        rows = []
        for t, direction in evs:
            start = np.searchsorted(ts, t + lag_ms * 1000)
            if start >= len(ts):
                continue
            entry = mid[start]
            values = []
            for h in horizons_s:
                end = np.searchsorted(ts, ts[start] + int(h * 1e6))
                if end >= len(ts):
                    values.append(np.nan)
                else:
                    values.append((mid[end] / entry - 1) * 1e4 * direction)
            rows.append(values)
        out[name] = np.array(rows) if rows else np.empty((0, len(horizons_s)))
    return out


def baseline(mids, horizons_s, lag_ms, count=4000, seed=0):
    """Безусловное среднее: те же горизонты со случайных моментов."""
    ts = np.array([t for t, _ in mids], dtype=np.int64)
    mid = np.array([m for _, m in mids])
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(ts) - 1, size=min(count, len(ts) - 1), replace=False)
    rows = []
    for i in picks:
        entry = mid[i]
        values = []
        for h in horizons_s:
            end = np.searchsorted(ts, ts[i] + int(h * 1e6))
            values.append((mid[end] / entry - 1) * 1e4 if end < len(ts) else np.nan)
        rows.append(values)
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTC_USDT")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--lag-ms", type=int, default=200)
    ap.add_argument("--absorb-k", type=float, default=2.0,
                    help="во сколько раз проторгованное превышает показанное")
    ap.add_argument("--absorb-window", type=float, default=10.0)
    ap.add_argument("--sweep-bp", type=float, default=3.0)
    ap.add_argument("--reclaim-s", type=float, default=10.0)
    ap.add_argument("--wall-size", type=float, default=50000)
    ap.add_argument("--wall-near-bp", type=float, default=5.0)
    a = ap.parse_args()

    cfg = {"absorb_k": a.absorb_k, "absorb_window_s": a.absorb_window,
           "sweep_bp": a.sweep_bp, "reclaim_s": a.reclaim_s,
           "wall_size": a.wall_size, "wall_near_bp": a.wall_near_bp}
    horizons = [5, 30, 60, 300]

    mids, events = collect(a.symbol, a.hours, cfg)
    if len(mids) < 1000:
        sys.exit("Слишком мало данных.")
    res = forward(mids, events, horizons, a.lag_ms)
    base = baseline(mids, horizons, a.lag_ms)

    span_h = (mids[-1][0] - mids[0][0]) / 1e6 / 3600
    print("=" * 88)
    print(f"  {a.symbol}   {span_h:.1f} ч   задержка решения {a.lag_ms} мс")
    print(f"  издержки круга: maker/maker 0, maker/taker ~1.0 б.п.")
    print("=" * 88)
    print(f"\n  {'событие':<20}{'штук':>7}{'в сутки':>9}"
          + "".join(f"{f'{h}с':>18}" for h in horizons))
    print(f"  {'':<20}{'':>7}{'':>9}"
          + "".join(f"{'ср.б.п.':>10}{'t':>8}" for _ in horizons))
    print("  " + "-" * 84)

    for name, arr in res.items():
        row = f"  {name:<20}{len(arr):>7}{len(arr)/max(span_h,1)*24:>9.1f}"
        for i in range(len(horizons)):
            if len(arr) < 5:
                row += f"{'—':>10}{'—':>8}"
                continue
            col = arr[:, i]
            col = col[np.isfinite(col)]
            if len(col) < 5 or col.std() == 0:
                row += f"{'—':>10}{'—':>8}"
                continue
            t = col.mean() / (col.std(ddof=1) / np.sqrt(len(col)))
            row += f"{col.mean():>10.3f}{t:>8.2f}"
        print(row)

    row = f"  {'СЛУЧАЙНЫЙ МОМЕНТ':<20}{len(base):>7}{'—':>9}"
    for i in range(len(horizons)):
        col = base[:, i]
        col = col[np.isfinite(col)]
        t = col.mean() / (col.std(ddof=1) / np.sqrt(len(col))) if len(col) > 5 else 0
        row += f"{col.mean():>10.3f}{t:>8.2f}"
    print(row)

    print("=" * 88)
    print("  Событие имеет смысл, только если ср.б.п. заметно больше издержек")
    print("  и t выше 3. Сравнивать надо со строкой случайного момента: если")
    print("  событие её не обыгрывает, это просто момент времени, а не сетап.")
    print("=" * 88)


if __name__ == "__main__":
    main()
