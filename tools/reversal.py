#!/usr/bin/env python3
"""Разворачивается ли цена после резкого движения — на всей выборке.

    python tools/reversal.py --symbol all

Та же гипотеза, что и «свип с возвратом», но без детектора событий.

Почему без него. Событийный тест оказался бессилен: строгое определение свипа
оставляет 4-18 событий в сутки, и на сорока часах записи это 7-31 наблюдение
на инструмент. При таком размере выборки t не поднимается выше единицы, а
«победитель» скачет между инструментами от одной поправки к другой — то есть
метод не различает гипотезу и шум.

Здесь то же самое меряется непрерывно: для КАЖДОГО момента времени берётся
движение за прошлую секунду и доходность за следующие 30-300 секунд.
Наблюдений становится сотни тысяч вместо десятков.

Движение меряется в собственных сигмах инструмента, а не в базисных пунктах —
единица должна быть естественной для рынка, а не для наблюдателя.

Знак: положительная «отработка» означает, что ФЕЙД работает (цена
возвращается), отрицательная — что работает продолжение движения.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.analyze import grid
from tools.setups import block_t

BUCKETS = [(-99, -5), (-5, -3), (-3, -2), (-2, -1), (-1, 1),
           (1, 2), (2, 3), (3, 5), (5, 8), (8, 99)]


def study(symbol, hours, step_ms, lag_ms, horizons, use_cache=True):
    g = grid(symbol, step_ms, hours, use_cache=use_cache)
    ts, mid, seg = g["ts"], g["mid"], g["seg"]
    n = len(mid)
    if n < 5000:
        return None

    back = max(1, 1000 // step_ms)          # окно прошлого движения: 1 секунда
    lag = max(1, lag_ms // step_ms)

    past = np.full(n, np.nan)
    ok = seg[back:] == seg[:-back]
    past[back:] = np.where(ok, (mid[back:] / mid[:-back] - 1) * 1e4, np.nan)
    # Именно стандартное отклонение, а не медиана: на BTC цена больше
    # половины секунд стоит на месте, и медиана |движения| равна нулю.
    sigma = float(np.nanstd(past[np.isfinite(past)]))
    if not sigma or not np.isfinite(sigma) or sigma <= 0:
        return None
    past_sig = past / sigma

    out = {"symbol": symbol, "sigma_1s_bp": float(sigma), "n": n,
           "hours": float((ts[-1] - ts[0]) / 1e6 / 3600), "rows": []}

    for h_s in horizons:
        h = max(1, int(h_s * 1000 / step_ms))
        end = n - lag - h
        if end <= 1000:
            continue
        fut = np.full(n, np.nan)
        same = seg[lag + h: lag + h + end] == seg[lag: lag + end]
        fut[:end] = np.where(
            same, (mid[lag + h: lag + h + end] / mid[lag: lag + end] - 1) * 1e4,
            np.nan)
        # ДРЕЙФ УБИРАЕМ. Метрика «-знак(прошлого) x будущее» устроена так, что
        # любой устойчивый снос цены выглядит как работающий фейд после
        # движений вверх и работающее продолжение после движений вниз. Именно
        # это и вышло на первом прогоне: знак переворачивался между половинами
        # записи почти на всех инструментах, потому что переворачивался дрейф.
        # Вычитаем среднюю доходность внутри получасовых блоков: снос медленнее
        # получаса уходит, эффект на горизонте 30-300 секунд остаётся.
        block = ts // (1800 * 1_000_000)
        fut_d = fut.copy()
        finite = np.isfinite(fut)
        for b in np.unique(block[finite]):
            m = finite & (block == b)
            if m.sum() > 20:
                fut_d[m] = fut[m] - fut[m].mean()
        fade = -np.sign(past_sig) * fut_d
        for lo, hi in BUCKETS:
            mask = (past_sig >= lo) & (past_sig < hi) & np.isfinite(fade)
            if mask.sum() < 200:
                continue
            values, stamps = fade[mask], ts[mask]
            t_val, blocks = block_t(values, stamps)
            cut = stamps.min() + (stamps.max() - stamps.min()) / 2
            h1 = values[stamps < cut]
            h2 = values[stamps >= cut]
            # средние по получасовым блокам — из них потом собирается
            # объединённая проверка по всем инструментам
            keys = stamps // (1800 * 1_000_000)
            per_block = {}
            for k, v in zip(keys, values):
                per_block.setdefault(int(k), []).append(v)
            out["rows"].append({
                "h": h_s, "lo": lo, "hi": hi, "n": int(mask.sum()),
                "mean": float(values.mean()), "t": t_val, "blocks": blocks,
                "h1": float(h1.mean()) if len(h1) > 50 else float("nan"),
                "h2": float(h2.mean()) if len(h2) > 50 else float("nan"),
                "block_means": {k: float(np.mean(v)) for k, v in per_block.items()
                                if len(v) >= 5}})
    return out


def show(res, only_h=None):
    print(f"\n  {res['symbol']}   {res['hours']:.0f} ч   узлов {res['n']:,}   "
          f"сигма движения за 1 с = {res['sigma_1s_bp']:.3f} б.п.")
    print(f"    {'движение':<14}{'гор.':>6}{'наблюдений':>12}{'фейд б.п.':>11}"
          f"{'t':>7}{'1-я пол.':>10}{'2-я пол.':>10}")
    for r in res["rows"]:
        if only_h and r["h"] != only_h:
            continue
        name = (f"{r['lo']:+.0f}..{r['hi']:+.0f} сигм" if abs(r["lo"]) < 90
                else f"< {r['hi']:+.0f} сигм")
        if r["lo"] > 90 or r["hi"] > 90:
            name = f"> {r['lo']:+.0f} сигм"
        mark = ""
        if abs(r["t"]) > 3 and np.sign(r["h1"]) == np.sign(r["h2"]):
            mark = "  <<<" if r["mean"] > 0 else "  <<< (продолжение)"
        print(f"    {name:<14}{r['h']:>6}{r['n']:>12,}{r['mean']:>11.3f}"
              f"{r['t']:>7.2f}{r['h1']:>10.3f}{r['h2']:>10.3f}{mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTC_USDT")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--step-ms", type=int, default=200)
    ap.add_argument("--lag-ms", type=int, default=200)
    ap.add_argument("--no-cache", action="store_true",
                    help="пересчитать сетку заново, не брать из кэша")
    ap.add_argument("--horizon", type=int, default=60,
                    help="какой горизонт показывать в сводке")
    a = ap.parse_args()

    from config import SYMBOLS
    targets = SYMBOLS if a.symbol == "all" else [a.symbol]
    horizons = [30, 60, 300]

    print("=" * 96)
    print("  РАЗВОРОТ ПОСЛЕ РЕЗКОГО ДВИЖЕНИЯ")
    print("  Движение за прошлую секунду — в собственных сигмах инструмента")
    print("  (стандартное отклонение односекундных изменений).")
    print("  «Фейд б.п.» > 0 значит цена возвращается, < 0 — движение продолжается.")
    print("  Дрейф цены вычтен по получасовым блокам: без этого обычный снос")
    print("  инструмента неотличим от разворота.")
    from tools.backtest import taker_fee_bp
    fees = {sym: taker_fee_bp(sym) for sym in targets}
    round_bp = 2 * max(fees.values())
    print("  Издержки круга по рынку: комиссия у инструментов РАЗНАЯ — "
          + ", ".join(f"{s.split('_')[0]} {f:g}" for s, f in fees.items())
          + " б.п. за сторону.")
    print(f"  Ниже сверяемся с худшей: {round_bp:g} б.п. за круг.")
    print("=" * 96)

    results = []
    for sym in targets:
        try:
            res = study(sym, a.hours, a.step_ms, a.lag_ms, horizons,
                        use_cache=not a.no_cache)
        except SystemExit:
            res = None
        if not res:
            print(f"  {sym}: данных мало")
            continue
        results.append(res)
        show(res, only_h=a.horizon if len(targets) > 1 else None)

    if len(results) > 1:
        print("\n" + "=" * 96)
        print(f"  СВОДКА ПО КРАЙНЕМУ ВЕДРУ (движение больше 3 сигм, "
              f"горизонт {a.horizon} с)")
        print("=" * 96)
        print(f"  {'инструмент':<14}{'наблюдений':>12}{'фейд б.п.':>11}{'t':>8}"
              f"{'1-я пол.':>10}{'2-я пол.':>10}")
        agree = 0
        for res in results:
            row = next((r for r in res["rows"]
                        if r["h"] == a.horizon and r["lo"] >= 3), None)
            if not row:
                print(f"  {res['symbol']:<14}{'—':>12}")
                continue
            if abs(row["t"]) > 3 and np.sign(row["h1"]) == np.sign(row["h2"]):
                agree += 1
            print(f"  {res['symbol']:<14}{row['n']:>12,}{row['mean']:>11.3f}"
                  f"{row['t']:>8.2f}{row['h1']:>10.3f}{row['h2']:>10.3f}")
        print("=" * 96)
        print(f"  Значимо и с совпадающим знаком на половинах: {agree} из {len(results)}")

        # --- объединение инструментов ------------------------------------
        # Складывать t-статистики нельзя: инструменты ходят вместе, их ошибки
        # коррелируют, и сумма завысит уверенность. Поэтому сначала усредняем
        # инструменты ВНУТРИ каждого получасового блока — общее движение рынка
        # схлопывается в одно наблюдение, — и только потом считаем t по блокам.
        pooled = {}
        for res in results:
            row = next((r for r in res["rows"]
                        if r["h"] == a.horizon and r["lo"] >= 3), None)
            if not row:
                continue
            for k, v in row["block_means"].items():
                pooled.setdefault(k, []).append(v)
        shared = [np.mean(v) for v in pooled.values() if len(v) >= 3]
        if len(shared) >= 10:
            arr_p = np.array(shared)
            t_p = arr_p.mean() / (arr_p.std(ddof=1) / np.sqrt(len(arr_p)))
            half = len(arr_p) // 2
            print()
            print("  ОБЪЕДИНЁННАЯ ПРОВЕРКА (инструменты усреднены внутри")
            print("  получасовых блоков — так учтено, что они ходят вместе)")
            print(f"    блоков {len(arr_p)}   среднее {arr_p.mean():+.3f} б.п.   "
                  f"t = {t_p:.2f}")
            print(f"    первая половина {arr_p[:half].mean():+.3f}   "
                  f"вторая {arr_p[half:].mean():+.3f}")
            print(f"    издержки круга по худшей ставке {round_bp:g} б.п. — "
                  f"{'перекрывается' if arr_p.mean() > round_bp else 'НЕ перекрывается'}")
        print("  Знак должен быть ОДИНАКОВЫМ на всех инструментах — иначе это не")
        print("  свойство стакана, а особенности отдельных рынков или шум.")
        print("=" * 96)


if __name__ == "__main__":
    main()
