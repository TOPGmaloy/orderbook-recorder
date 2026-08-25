#!/usr/bin/env python3
"""Существует ли вообще прибыльная конструкция сделки — или их нет.

    python tools/sweep.py --symbol BTC_USDT --hours 24

Это НЕ подбор параметров. Подбор здесь бессмыслен и вреден: настроенный на
одном куске результат не переносится на следующий, это уже проверялось на
моментум-стратегиях. Здесь проверяется другое — структурный вопрос.

Арифметика такая: измеренное преимущество около 0.35 б.п. на сделку, а любой
выход по рынку стоит 1.0 б.п. Значит конструкция, где заметная доля выходов
рыночные, убыточна независимо от качества сигнала. Вопрос: бывает ли
конструкция, где выходы преимущественно лимитные и при этом сделки вообще
случаются.

Поэтому в таблице главный столбец — не прибыль, а ДОЛЯ ЛИМИТНЫХ ВЫХОДОВ.
Прибыль без неё смысла не имеет.

Каждая конструкция считается на обеих половинах записи отдельно. Совпадение
знака на двух половинах — минимальное требование; несовпадение означает шум,
какой бы красивой ни была общая цифра.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.backtest import replay_multi, strategy, random_control

# Конструкции выбраны из измеренной динамики, а не перебором:
# медиана движения за 60 с — 2.155 б.п., поэтому стоп в 2 б.п. стоит внутри
# шума; 999 означает «стопа нет», выход только по цели или по таймеру.
GRID = [
    #  имя                          стоп  цель  таймер  порог
    ("стоп 2 / цель 2 / 60с",         2,    2,     60,   0.4),
    ("стоп 6 / цель 3 / 60с",         6,    3,     60,   0.4),
    ("стоп 8 / цель 6 / 120с",        8,    6,    120,   0.4),
    ("без стопа / цель 2 / 30с",    999,    2,     30,   0.4),
    ("без стопа / цель 3 / 60с",    999,    3,     60,   0.4),
    ("без стопа / цель 5 / 300с",   999,    5,    300,   0.4),
    ("сильный сигнал, без стопа",   999,    3,     60,   0.7),
    ("сильный сигнал, стоп 6",        6,    3,     60,   0.7),
]


def split_stats(trades, cutoff_us):
    """Первая и вторая половина записи считаются раздельно."""
    out = []
    for chunk in ([t for t in trades if t.entry_us < cutoff_us],
                  [t for t in trades if t.entry_us >= cutoff_us]):
        if len(chunk) < 2:
            out.append(None)
            continue
        x = np.array([t.pnl_bp for t in chunk])
        t_stat = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if x.std() else 0.0
        out.append({"n": len(x), "mean": float(x.mean()), "t": float(t_stat)})
    return out


def maker_share(trades):
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.exit_reason == "цель лимиткой") / len(trades) * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTC_USDT")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--lag-ms", type=int, default=200)
    ap.add_argument("--taker-bp", type=float, default=1.0)
    ap.add_argument("--notional", type=float, default=2500)
    a = ap.parse_args()

    base = {"hours": a.hours, "lag_ms": a.lag_ms, "step_ms": 200,
            "taker_bp": a.taker_bp, "size": 1.0, "order_life_s": 10,
            "stop_bp": 2, "target_bp": 2, "time_stop_s": 60,
            "imb_th": 0.4, "ofi_th": 0.0}

    runs = []
    for name, stop, target, timer, th in GRID:
        params = dict(base, stop_bp=stop, target_bp=target,
                      time_stop_s=timer, imb_th=th)
        runs.append((name, params, strategy(params)))
    control = dict(base, stop_bp=999, target_bp=3, time_stop_s=60)
    runs.append(("СЛУЧАЙНЫЙ ВХОД", control, random_control(0.01)))

    print("=" * 96)
    print(f"  {a.symbol}   последние {a.hours:g} ч   задержка {a.lag_ms} мс   "
          f"выход по рынку {a.taker_bp} б.п.   номинал ${a.notional:,.0f}")
    print("=" * 96)

    result = replay_multi(a.symbol, base, runs)

    all_trades = [t for trades, _ in result.values() for t in trades]
    if not all_trades:
        print("  ни одна конструкция не совершила сделок")
        return
    lo = min(t.entry_us for t in all_trades)
    hi = max(t.entry_us for t in all_trades)
    cutoff = lo + (hi - lo) / 2

    print(f"\n  {'конструкция':<28}{'сделок':>7}{'лимитных':>10}"
          f"{'1-я половина':>22}{'2-я половина':>22}")
    print(f"  {'':<28}{'':>7}{'выходов':>10}"
          f"{'ср.б.п.':>11}{'t':>11}{'ср.б.п.':>11}{'t':>11}")
    print("  " + "-" * 92)
    for name, _, _ in runs:
        trades, _ = result[name]
        first, second = split_stats(trades, cutoff)
        row = f"  {name:<28}{len(trades):>7}{maker_share(trades):>9.0f}%"
        for half in (first, second):
            row += (f"{half['mean']:>11.3f}{half['t']:>11.2f}"
                    if half else f"{'—':>11}{'—':>11}")
        print(row)

    print("=" * 96)
    print("  Читать так: конструкция чего-то стоит, только если ср.б.п. положительно")
    print("  на ОБЕИХ половинах и обе t выше 3. Плюс на одной половине и минус на")
    print("  другой — это шум. Доля лимитных выходов ниже 70% означает, что комиссия")
    print("  съест преимущество независимо от остальных цифр.")
    print("=" * 96)


if __name__ == "__main__":
    main()
