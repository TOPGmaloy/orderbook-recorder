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
            events["поглощение"].append(
                (ts, direction, {"k": st["traded"] / st["shown"]}))
            st["traded"] = 0.0
            st["since"] = ts

        # --- СВИП С ВОЗВРАТОМ ----------------------------------------------
        window = [m for t, m in recent_mid if ts - t <= 1_000_000]
        if len(window) > 3:
            lo, hi = min(window), max(window)
            span_bp = (hi - lo) / mid * 1e4
            if swept is None and span_bp >= cfg["sweep_bp"]:
                direction = -1 if mid >= (hi + lo) / 2 else +1   # ждём возврата
                swept = (ts, direction, (hi + lo) / 2, span_bp)
            elif swept is not None:
                age = (ts - swept[0]) / 1e6
                back = abs(mid - swept[2]) / mid * 1e4
                if age <= cfg["reclaim_s"] and back <= cfg["sweep_bp"] * 0.4:
                    events["свип с возвратом"].append(
                        (ts, swept[1],
                         {"span": swept[3], "back_ratio": back / swept[3]}))
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
            events["осушение"].append(
                (ts, direction, {"size": st["shown"]}))
            level_state.pop(price, None)

        # чистим состояние далёких уровней
        if len(level_state) > 4000:
            level_state = {p: s for p, s in level_state.items()
                           if abs(p - mid) / mid * 1e4 < 50}

    return mids, events


def block_t(values, stamps, block_s=300):
    """t по блокам времени, а не по отдельным событиям.

    События кучкуются: в волатильные минуты их десятки подряд, и доходности
    после них пересекаются. Обычный t-критерий на таких данных завышен, иногда
    вдвое. Здесь события сначала усредняются внутри пятиминутных блоков, и t
    считается по блокам — соседние блоки уже почти независимы.
    """
    if len(values) < 10:
        return float("nan"), 0
    blocks = {}
    for v, t in zip(values, stamps):
        blocks.setdefault(int(t // (block_s * 1_000_000)), []).append(v)
    means = np.array([np.mean(v) for v in blocks.values()])
    if len(means) < 5 or means.std() == 0:
        return float("nan"), len(means)
    return float(means.mean() / (means.std(ddof=1) / np.sqrt(len(means)))), len(means)


def forward(mids, events, horizons_s, lag_ms):
    ts = np.array([t for t, _ in mids], dtype=np.int64)
    mid = np.array([m for _, m in mids])
    out = {}
    for name, evs in events.items():
        rows, stamps, meta = [], [], []
        for t, direction, info in evs:
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
            rows.append(values); stamps.append(t); meta.append(info)
        out[name] = (np.array(rows) if rows else np.empty((0, len(horizons_s))),
                     np.array(stamps, dtype=np.int64), meta)
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


def stat(arr, stamps, col):
    """Среднее, блочная t и половинки записи по одному горизонту."""
    if len(arr) < 10:
        return None
    values = arr[:, col]
    good = np.isfinite(values)
    values, times = values[good], stamps[good]
    if len(values) < 10:
        return None
    t_all, blocks = block_t(values, times)
    cut = times.min() + (times.max() - times.min()) / 2
    halves = []
    for mask in (times < cut, times >= cut):
        halves.append(float(values[mask].mean()) if mask.sum() >= 5 else float("nan"))
    return {"n": len(values), "mean": float(values.mean()), "t": t_all,
            "blocks": blocks, "first": halves[0], "second": halves[1]}


def sweep_scan(symbol, arr, stamps, meta, span_h, compact=False):
    """Таблица «сила свипа против отработки». Главная проверка на подлинность."""
    if len(arr) < 20:
        print(f"  {symbol:<12} событий мало ({len(arr)}), сравнивать нечего")
        return None
    spans = np.array([m["span"] for m in meta])
    best = None
    rows = []
    for th in (1.5, 2.5, 4.0, 6.0, 9.0, 13.0, 20.0):
        mask = spans >= th
        if mask.sum() < 6:
            continue
        sub, sub_ts = arr[mask][:, 2], stamps[mask]
        good = np.isfinite(sub)
        sub, sub_ts = sub[good], sub_ts[good]
        if len(sub) < 6:
            continue
        t_val, _ = block_t(sub, sub_ts) if len(sub) >= 10 else (float("nan"), 0)
        cut = sub_ts.min() + (sub_ts.max() - sub_ts.min()) / 2
        h1 = sub[sub_ts < cut]
        h2 = sub[sub_ts >= cut]
        rows.append({"th": th, "n": len(sub), "per_day": len(sub) / max(span_h, 1) * 24,
                     "mean": float(sub.mean()), "t": t_val,
                     "h1": float(h1.mean()) if len(h1) >= 3 else float("nan"),
                     "h2": float(h2.mean()) if len(h2) >= 3 else float("nan")})
    if not rows:
        return None
    strict = rows[-1]
    grows = len(rows) >= 2 and rows[-1]["mean"] > rows[0]["mean"]
    same_sign = (np.sign(strict["h1"]) == np.sign(strict["h2"])
                 and np.isfinite(strict["h1"]) and np.isfinite(strict["h2"]))
    if compact:
        mark = ""
        if grows and same_sign and strict["mean"] > 1.0:
            mark = "  <-- растёт, знак совпал"
        print(f"  {symbol:<12}{strict['th']:>7.1f}{strict['n']:>8}"
              f"{strict['per_day']:>9.0f}{strict['mean']:>10.2f}{strict['t']:>7.2f}"
              f"{strict['h1']:>9.2f}{strict['h2']:>9.2f}{mark}")
    else:
        print(f"    {'порог':<8}{'событий':>8}{'/сутки':>8}"
              f"{'60с ср.':>9}{'t':>6}{'1-я пол.':>10}{'2-я пол.':>10}")
        for r in rows:
            print(f"    {r['th']:<8.1f}{r['n']:>8}{r['per_day']:>8.0f}"
                  f"{r['mean']:>9.3f}{r['t']:>6.2f}{r['h1']:>10.2f}{r['h2']:>10.2f}")
    return {"symbol": symbol, "grows": grows, "same_sign": same_sign, **strict}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTC_USDT")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--lag-ms", type=int, default=200)
    ap.add_argument("--absorb-k", type=float, default=1.5)
    ap.add_argument("--absorb-window", type=float, default=10.0)
    ap.add_argument("--sweep-bp", type=float, default=1.5)
    ap.add_argument("--reclaim-s", type=float, default=10.0)
    ap.add_argument("--wall-size", type=float, default=30000)
    ap.add_argument("--wall-near-bp", type=float, default=5.0)
    a = ap.parse_args()

    from config import SYMBOLS
    if a.symbol == "all":
        print("=" * 92)
        print("  СВИП С ВОЗВРАТОМ ПО ВСЕМ ИНСТРУМЕНТАМ — независимые выборки")
        print("  Если это механизм стакана, он проявится не только на BTC.")
        print("=" * 92)
        print(f"  {'инструмент':<12}{'порог':>7}{'событий':>8}{'/сутки':>9}"
              f"{'60с ср.':>10}{'t':>7}{'1-я пол.':>9}{'2-я пол.':>9}")
        found = []
        for sym in SYMBOLS:
            cfg_all = {"absorb_k": 99, "absorb_window_s": a.absorb_window,
                       "sweep_bp": a.sweep_bp, "reclaim_s": a.reclaim_s,
                       "wall_size": 1e18, "wall_near_bp": a.wall_near_bp}
            mids_s, ev = collect(sym, a.hours, cfg_all)
            if len(mids_s) < 1000:
                print(f"  {sym:<12} данных мало")
                continue
            r = forward(mids_s, ev, [5, 30, 60, 300], a.lag_ms)
            arr, stamps, meta = r["свип с возвратом"]
            span = (mids_s[-1][0] - mids_s[0][0]) / 1e6 / 3600
            out = sweep_scan(sym, arr, stamps, meta, span, compact=True)
            if out:
                found.append(out)
        good = [f for f in found if f["grows"] and f["same_sign"] and f["mean"] > 1.0]
        print("=" * 92)
        print(f"  Условие выполнено на {len(good)} инструментах из {len(found)}:"
              f" {', '.join(f['symbol'] for f in good) or '—'}")
        print("  Одно совпадение — случайность. Совпадение на трёх и более")
        print("  инструментах с разным потоком — уже свойство механизма.")
        print("=" * 92)
        return

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

    print("=" * 92)
    print(f"  {a.symbol}   {span_h:.1f} ч   задержка решения {a.lag_ms} мс")
    print("  t считается ПО БЛОКАМ времени: события кучкуются, и обычный")
    print("  t-критерий на них завышен. Порог значимости прежний — 3.")
    print("=" * 92)

    for name, (arr, stamps, meta) in res.items():
        print(f"\n  {name.upper()}   событий {len(arr)}, "
              f"{len(arr)/max(span_h,1)*24:.0f} в сутки")
        if len(arr) < 10:
            print("    слишком мало для выводов")
            continue
        print(f"    {'горизонт':<10}{'ср.б.п.':>10}{'t':>8}{'блоков':>8}"
              f"{'1-я пол.':>10}{'2-я пол.':>10}")
        for i, h in enumerate(horizons):
            st = stat(arr, stamps, i)
            if not st:
                continue
            mark = "  <<<" if st["t"] > 3 and st["mean"] > 1.0 else ""
            print(f"    {str(h)+' с':<10}{st['mean']:>10.3f}{st['t']:>8.2f}"
                  f"{st['blocks']:>8}{st['first']:>10.3f}{st['second']:>10.3f}{mark}")

    print(f"\n  СЛУЧАЙНЫЙ МОМЕНТ   {len(base)} точек")
    print(f"    {'горизонт':<10}{'ср.б.п.':>10}")
    for i, h in enumerate(horizons):
        col = base[:, i][np.isfinite(base[:, i])]
        print(f"    {str(h)+' с':<10}{col.mean():>10.3f}")

    # --- скан по силе свипа ------------------------------------------------
    arr, stamps, meta = res.get("свип с возвратом", (np.empty((0, 4)), None, []))
    if len(arr) >= 30:
        print("\n" + "=" * 92)
        print("  СКАН ПО СИЛЕ СВИПА — главная проверка на подлинность")
        print("  Если преимущество РАСТЁТ со строгостью порога, эффект настоящий.")
        print("  Если не растёт или падает — мы ловим шум, а не событие.")
        print("=" * 92)
        spans = np.array([m["span"] for m in meta])
        print(f"    {'порог':<8}{'событий':>8}{'/сутки':>8}"
              f"{'30с ср.':>9}{'t':>6}{'60с ср.':>9}{'t':>6}"
              f"{'60с 1-я':>9}{'60с 2-я':>9}")
        for th in (1.5, 2.5, 4.0, 6.0, 9.0, 13.0, 20.0):
            mask = spans >= th
            if mask.sum() < 8:
                continue
            row = (f"    {th:<8.1f}{int(mask.sum()):>8}"
                   f"{mask.sum()/max(span_h,1)*24:>8.0f}")
            for hi in (1, 2):
                sub, sub_ts = arr[mask][:, hi], stamps[mask]
                good = np.isfinite(sub)
                if good.sum() < 8:
                    row += f"{'—':>9}{'—':>6}"
                    continue
                t_val, _ = block_t(sub[good], sub_ts[good])
                row += f"{sub[good].mean():>9.3f}{t_val:>6.2f}"
            # половины записи по горизонту 60 с: знак обязан совпасть
            sub, sub_ts = arr[mask][:, 2], stamps[mask]
            good = np.isfinite(sub)
            sub, sub_ts = sub[good], sub_ts[good]
            if len(sub) >= 8:
                cut = sub_ts.min() + (sub_ts.max() - sub_ts.min()) / 2
                for m in (sub_ts < cut, sub_ts >= cut):
                    row += (f"{sub[m].mean():>9.2f}" if m.sum() >= 3
                            else f"{'—':>9}")
            print(row)

    print("\n" + "=" * 92)
    print("  Сетап годится, если: преимущество заметно больше 1 б.п., блочная t")
    print("  выше 3, знак совпадает на обеих половинах записи, и преимущество")
    print("  растёт со строгостью порога. Иначе это момент времени, а не сетап.")
    print("=" * 92)



if __name__ == "__main__":
    main()
