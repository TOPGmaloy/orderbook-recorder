#!/usr/bin/env python3
"""Чем рискует плечо: как далеко цена уходит ПРОТИВ позиции за время удержания.

    python tools/tails.py --symbol HYPE_USDT --hold-s 60

Это замер риска инструмента, а не тест стратегии: замороженное правило он не
трогает и ничего в нём не подбирает.

Считается не доходность на выходе, а ХУДШИЙ ход против позиции за время
удержания. Разница принципиальная: ликвидация не ждёт закрытия сделки. Позиция
может выйти в плюс к концу минуты, но если в середине минуты цена сходила
против нас глубже, чем позволяет маржа, счёта уже нет.

Порог ликвидации при плече N — примерно (1/N − поддерживающая маржа). При 20x
это около 4.5%, при 50x около 1.5%. Здесь считается, как часто такой ход
вообще случался в записи — в те самые моменты, когда правило входит в рынок,
и с той стороной, в которую оно входит.

ЧЕГО ЭТОТ ЗАМЕР НЕ МОЖЕТ. Записи несколько суток. Хвост, случающийся раз в год,
в неё не попал и попасть не мог. Поэтому все числа ниже — НИЖНЯЯ оценка риска:
настоящий хвост толще, и насколько — по этим данным не узнать никак.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.analyze import build_grids, rolling
from tools.backtest import contract_detail
from tools.scale import rolling_sigma

LEVERAGE = (1, 3, 5, 10, 20, 50, 100)


def sliding_extreme(x, w, biggest):
    """Минимум или максимум в окне длины w, начинающемся в каждом узле.

    Наивный проход по окну — это n*w операций, на полутора миллионах узлов и
    окне в триста узлов он считался бы минутами. Здесь блочный приём: массив
    режется на блоки длины w, внутри блоков берутся накопленные минимумы
    слева и справа, и любое окно собирается из двух готовых кусков.
    """
    take = np.maximum if biggest else np.minimum
    fill = -np.inf if biggest else np.inf
    n = len(x)
    pad = (-n) % w
    xp = np.concatenate([x, np.full(pad, fill)])
    blocks = xp.reshape(-1, w)
    prefix = take.accumulate(blocks, axis=1).ravel()
    suffix = take.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].ravel()
    # окно, ЗАКАНЧИВАЮЩЕЕСЯ в i
    ending = np.full(len(xp), fill)
    idx = np.arange(w - 1, len(xp))
    ending[idx] = take(suffix[idx - w + 1], prefix[idx])
    # нам нужно окно, НАЧИНАЮЩЕЕСЯ в i
    out = np.full(n, np.nan)
    good = n - w + 1
    if good > 0:
        out[:good] = ending[w - 1: w - 1 + good]
    return out


def _check_sliding():
    """Сверка блочного приёма с прямым перебором на мелком массиве.

    Ошибка на единицу в индексах здесь не падает, а тихо сдвигает окно — и
    риск оказывается посчитан не по тому куску.
    """
    rng = np.random.default_rng(3)
    for n, w in ((37, 5), (100, 7), (64, 8), (50, 1)):
        x = rng.normal(size=n)
        for biggest in (False, True):
            got = sliding_extreme(x, w, biggest)
            for i in range(n - w + 1):
                want = x[i:i + w].max() if biggest else x[i:i + w].min()
                assert abs(got[i] - want) < 1e-12, (n, w, biggest, i)
    return True


def maintenance_rate(symbol):
    row = contract_detail().get(symbol) or {}
    try:
        return float(row.get("maintenanceMarginRate") or 0.005)
    except (TypeError, ValueError):
        return 0.005


def measure(symbol, g, a):
    step, hold = a.step_ms, max(1, int(a.hold_s * 1000) // a.step_ms)
    mid, seg = g["mid"], g["seg"]
    n = len(mid)
    if n < hold * 4:
        return None

    raw = rolling(g["delta_raw"], max(1, 5000 // step))
    scale = rolling_sigma(raw, step)
    with np.errstate(invalid="ignore", divide="ignore"):
        sig = np.where(np.isfinite(scale) & (scale > 0), raw / scale, np.nan)

    low = sliding_extreme(mid, hold, False)
    high = sliding_extreme(mid, hold, True)
    # позиция не должна переживать разрыв записи: если сегмент внутри окна
    # сменился, ход в этом окне мы не наблюдали
    same = np.full(n, False)
    same[:n - hold + 1] = seg[:n - hold + 1] == seg[hold - 1:]

    with np.errstate(invalid="ignore"):
        # лонг страдает от минимума, шорт от максимума; обе величины в б.п.
        adverse_long = (low / mid - 1) * 1e4
        adverse_short = (1 - high / mid) * 1e4
        adverse = np.where(sig > 0, adverse_long, adverse_short)

    entered = same & np.isfinite(adverse) & np.isfinite(sig) & (np.abs(sig) >= a.threshold)
    everywhere = same & np.isfinite(adverse_long)
    if entered.sum() < 100:
        return None
    return {"adverse": -adverse[entered],           # положительное = ход против нас
            "any": -np.minimum(adverse_long, adverse_short)[everywhere],
            "n": int(entered.sum()), "nodes": int(everywhere.sum())}


def show(symbol, res, a, per_day):
    mmr = maintenance_rate(symbol)
    adv = res["adverse"]
    print("\n" + "=" * 96)
    print(f"  {symbol}   удержание {a.hold_s:g} с   порог {a.threshold:g}σ   "
          f"поддерживающая маржа {mmr*100:.2f}%")
    print("=" * 96)
    q = np.percentile(adv, [50, 90, 99, 99.9])
    print(f"  ход ПРОТИВ позиции за удержание, {res['n']:,} входов: "
          f"медиана {q[0]:.1f} б.п., 90% {q[1]:.1f}, 99% {q[2]:.1f}, "
          f"99.9% {q[3]:.1f}, худший {adv.max():.1f}")
    print(f"  для сравнения, худший ход в любую сторону по всей записи: "
          f"{res['any'].max():.1f} б.п. ({res['any'].max()/100:.2f}%)")
    print()
    print(f"  {'плечо':>7}{'ликвидация при':>17}{'доля входов':>14}"
          f"{'как часто':>28}")
    for lev in LEVERAGE:
        threshold_bp = (1.0 / lev - mmr) * 1e4
        if threshold_bp <= 0:
            continue
        hits = int((adv >= threshold_bp).sum())
        share = hits / len(adv)
        if hits == 0:
            when = "ни разу за всю запись"
        else:
            days = 1.0 / (share * per_day)
            when = (f"раз в {days*24:.1f} ч" if days < 1
                    else f"раз в {days:.1f} сут")
        print(f"  {lev:>6}x{threshold_bp/100:>16.2f}%{share*100:>13.3f}%{when:>28}")
    print()
    print("  Числа — НИЖНЯЯ оценка: в записи несколько суток, и хвост, который")
    print("  случается раз в год, в неё попасть не мог. Настоящий риск больше.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="HYPE_USDT")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--step-ms", type=int, default=200)
    ap.add_argument("--hold-s", type=float, default=60,
                    help="сколько держим позицию: 60 для HYPE, 300 для XRP")
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--trades-per-day", type=float, default=510,
                    help="сколько сделок в сутки делает правило — из sweep")
    a = ap.parse_args()

    _check_sliding()
    print("=" * 96)
    print("  ЧЕМ РИСКУЕТ ПЛЕЧО")
    print("  Считается худший ход ПРОТИВ позиции за время удержания, а не")
    print("  результат на выходе: ликвидация не ждёт закрытия сделки.")
    print("  (скользящий минимум сверен с прямым перебором — ok)")
    print("=" * 96)

    from config import SYMBOLS
    targets = SYMBOLS if a.symbol == "all" else [a.symbol]
    grids = build_grids(targets, a.step_ms, a.hours)
    for symbol in targets:
        g = grids.get(symbol)
        if g is None:
            continue
        res = measure(symbol, g, a)
        if res is None:
            print(f"\n  {symbol}: данных мало")
            continue
        show(symbol, res, a, a.trades_per_day)
    print("\n" + "=" * 96)


if __name__ == "__main__":
    main()
