#!/usr/bin/env python3
"""Что остаётся от сигнала после издержек — по всем сигналам, а не по сотне сделок.

    python tools/edge.py --symbol all --hours 48

Зачем отдельно от sweep. Барьерная конструкция (стоп, цель, таймер) берёт
сигнал только тогда, когда движок свободен, — это единицы процентов сигналов,
— и добавляет к результату шум срабатывания барьеров. Разброс одной такой
сделки 9-30 б.п. при цели 8-25, поэтому чтобы увидеть преимущество в 1 б.п. с
t = 3, нужно от 700 до 8000 сделок на половину записи. В 48 часах их сотни.
Такой тест уверенно ловит минус в 3-5 б.п. — что он и делал, пока комиссия
была выдумана, — и не увидит плюс в 0.5-1 б.п., ровно тот, который решает дело
на инструментах без комиссии.

Здесь меряется прямо то, что нужно: средняя доходность входа ПО РЫНКУ в
сторону сигнала на фиксированном горизонте, минус издержки этого инструмента —
комиссия обеих сторон плюс спред, оба поузлово. Берутся ВСЕ узлы сетки, где
сигнал превысил порог: наблюдений на два порядка больше, барьеров нет, шума
меньше.

Что здесь может обмануть и как это закрыто:

  ПЕРЕКРЫТИЕ ОКОН. Соседние наблюдения на горизонте 5 минут — почти одно и то
  же наблюдение, и обычный t завышен в разы. Запись режется на блоки заведомо
  длиннее горизонта, внутри блока берётся среднее, t считается по блокам.
  Так же считает reversal.

  ЗАГЛЯДЫВАНИЕ ВПЕРЁД. Сигнал берётся в момент t, доходность считается от
  t + задержка. Масштаб сигнала — скользящий и причинный (tools/scale.py).

  НАПРАВЛЕНИЕ. Контроль — те же самые моменты, но сторона случайная. Он
  показывает, сколько стоит просто войти по рынку в это время; разница со
  строкой и есть вклад направления. Обгонять контроль обязательно: без этого
  положительное среднее означает лишь то, что рынок в эти часы рос.

  ПОЛОВИНЫ. Всё считается ещё и по половинам записи отдельно. Знак обязан
  совпасть — иначе это не преимущество, а период.

Столбец «времени» — доля узлов, где сигнал горит. Без неё матожидание на
сигнал не переводится в расписание сделок: сигналы идут пачками, а позицию
можно держать одну, и число сделок в сутки упирается не в число сигналов, а в
горизонт удержания.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.analyze import build_grids, rolling, FILL_NOTIONALS
from tools.backtest import taker_fee_bp, taker_fee_spread
from tools.scale import rolling_sigma

HORIZONS_MS = [60000, 300000, 900000]
THRESHOLDS = [2.0, 3.0, 4.0]
MIN_PER_BLOCK = 5          # блок из двух наблюдений — не среднее, а шум
MIN_BLOCKS = 8             # меньше блоков — считать t бессмысленно


def block_means(ts, values, ts0, block_us):
    """Средние по блокам времени: так учитывается перекрытие окон.

    Возвращает средние И номера блоков — номера нужны, чтобы делить запись на
    половины по времени, а не по числу наблюдений.
    """
    index = ((ts - ts0) // block_us).astype(np.int64)
    unique, inverse = np.unique(index, return_inverse=True)
    count = np.bincount(inverse)
    total = np.bincount(inverse, weights=values)
    keep = count >= MIN_PER_BLOCK
    return unique[keep], total[keep] / count[keep]


def t_stat(values):
    if len(values) < MIN_BLOCKS or np.std(values, ddof=1) == 0:
        return float("nan")
    return float(values.mean() / (np.std(values, ddof=1) / np.sqrt(len(values))))


def measure(symbol, g, a, fee):
    """Строки таблицы по одному инструменту."""
    step = a.step_ms
    lag = max(1, a.lag_ms // step)
    mid, seg, ts, spread = g["mid"], g["seg"], g["ts"], g["spread"]
    n = len(mid)

    raw = rolling(g["delta_raw"], max(1, 5000 // step))     # дельта сделок за 5 с
    scale = rolling_sigma(raw, step)
    with np.errstate(invalid="ignore", divide="ignore"):
        sig = np.where(np.isfinite(scale) & (scale > 0), raw / scale, np.nan)

    # цена исполнения выбранного объёма; на старых сетках её нет — тогда
    # считаем по спреду касания, как раньше, и говорим об этом в шапке
    buy_bp = g.get(f"buy{a.notional}")
    sell_bp = g.get(f"sell{a.notional}")

    rows = []
    for h_ms in HORIZONS_MS:
        h = max(1, h_ms // step)
        end = n - lag - h
        if end <= 1000:
            continue
        # доходность от момента входа (сигнал + задержка) до горизонта;
        # через разрывы записи не считаем
        same = seg[lag + h: lag + h + end] == seg[lag: lag + end]
        fut = np.full(n, np.nan)
        fut[:end] = np.where(
            same, (mid[lag + h: lag + h + end] / mid[lag: lag + end] - 1) * 1e4, np.nan)
        # Издержки круга: комиссия обеих сторон плюс НАСТОЯЩАЯ цена входа и
        # выхода по рынку для выбранного объёма. Спред касания — это цена
        # заявки, стремящейся к нулю; заявка на реальные деньги идёт по
        # стакану вглубь. Лонг покупает на входе и продаёт на выходе, шорт
        # наоборот, и книга на выходе своя — поэтому берутся четыре разных
        # узла, а не удвоенный спред.
        cost = np.full(n, np.nan)
        if buy_bp is None:
            cost[:end] = 2 * fee + (spread[lag: lag + end]
                                    + spread[lag + h: lag + h + end]) / 2
        else:
            long_cost = buy_bp[lag: lag + end] + sell_bp[lag + h: lag + h + end]
            short_cost = sell_bp[lag: lag + end] + buy_bp[lag + h: lag + h + end]
            cost[:end] = 2 * fee + np.where(sig[:end] > 0, long_cost, short_cost)

        block_us = max(1_800_000_000, 4 * h_ms * 1000)
        rng = np.random.default_rng(0)
        for th in THRESHOLDS:
            usable = np.isfinite(fut) & np.isfinite(sig)
            mask = usable & (np.abs(sig) >= th)
            share = mask.sum() / max(usable.sum(), 1) * 100
            if mask.sum() < 200:
                rows.append((h_ms, th, int(mask.sum()), 0,
                             float("nan"), float("nan"), float("nan"),
                             float("nan"), float("nan"), float("nan"), share))
                continue
            gross = np.sign(sig[mask]) * fut[mask]
            net = gross - cost[mask]
            signs = rng.choice([-1.0, 1.0], size=int(mask.sum()))
            ctrl = signs * fut[mask] - cost[mask]

            when = ts[mask]
            ids, net_b = block_means(when, net, ts[0], block_us)
            _, gross_b = block_means(when, gross, ts[0], block_us)
            _, ctrl_b = block_means(when, ctrl, ts[0], block_us)
            if len(net_b) < MIN_BLOCKS:
                rows.append((h_ms, th, int(mask.sum()), len(net_b),
                             float("nan"), float("nan"), float("nan"),
                             float("nan"), float("nan"), float("nan"), share))
                continue
            middle = (ids[0] + ids[-1]) / 2
            first, second = ids <= middle, ids > middle
            rows.append((h_ms, th, int(mask.sum()), len(net_b),
                         float(gross_b.mean()), float(net_b.mean()), t_stat(net_b),
                         float(net_b[first].mean()) if first.any() else float("nan"),
                         float(net_b[second].mean()) if second.any() else float("nan"),
                         float(ctrl_b.mean()), share))
    return rows


def depth_note(g, mask=None):
    """Во что обходится круг по рынку разного размера — медиана по записи."""
    out = []
    for notional in FILL_NOTIONALS:
        buy, sell = g.get(f"buy{notional}"), g.get(f"sell{notional}")
        if buy is None:
            continue
        both = buy + sell
        if mask is not None:
            both = both[mask]
        good = np.isfinite(both)
        if good.sum() < 100:
            out.append(f"${notional:,} стакан не держит")
            continue
        miss = (1 - good.mean()) * 100
        tail = f" (не хватало книги в {miss:.0f}% узлов)" if miss > 1 else ""
        out.append(f"${notional:,} -> {np.median(both[good]):.3f} б.п.{tail}")
    return "   ".join(out)


def miss_share(g, notional):
    """Доля узлов, где записанной книги не хватило на такой объём.

    Такие узлы из замера выпадают, и это не мелочь: книга не набирается там,
    где её только что пересобирали после разрыва, — то есть в самые рваные
    моменты. Если доля заметная, результат смещён в сторону спокойных минут,
    и знать об этом надо до, а не после.
    """
    both = g.get(f"buy{notional}")
    if both is None:
        return 0.0
    return float((~np.isfinite(both + g[f"sell{notional}"])).mean())


def show(symbol, rows, fee, spread_med):
    print("\n" + "=" * 100)
    note = " (БЕЗ КОМИССИИ)"
    if fee != 0:
        lo, hi = taker_fee_spread(symbol)
        note = (f" (биржа отвечала от {lo:g} до {hi:g} — взята худшая)"
                if lo is not None and hi != lo else "")
    print(f"  {symbol}   тейкер {fee:g} б.п.{note}   "
          f"медианный спред {spread_med:.3f} б.п.   "
          f"круг по рынку ≈ {2 * fee + spread_med:.3f} б.п.")
    print("=" * 100)
    print(f"  {'горизонт':<10}{'порог':>7}{'сигналов':>11}{'времени':>9}"
          f"{'блоков':>8}{'валовое':>10}{'ЧИСТОЕ':>10}{'t':>8}{'1-я пол.':>10}"
          f"{'2-я пол.':>10}{'контроль':>10}")
    print("  " + "-" * 105)
    for h_ms, th, n_sig, n_blk, gross, net, t, first, second, ctrl, share in rows:
        label = f"{h_ms/60000:g} мин" if h_ms >= 60000 else f"{h_ms/1000:g} с"
        if not np.isfinite(net):
            print(f"  {label:<10}{th:>6.0f}σ{n_sig:>11,}{share:>8.1f}%{n_blk:>8}"
                  f"{'мало данных':>48}")
            continue
        star = " <" if (net > 0 and first > 0 and second > 0 and t > 3) else ""
        print(f"  {label:<10}{th:>6.0f}σ{n_sig:>11,}{share:>8.1f}%{n_blk:>8}"
              f"{gross:>10.3f}{net:>10.3f}{t:>8.2f}{first:>10.3f}"
              f"{second:>10.3f}{ctrl:>10.3f}{star}")


def selftest():
    """Проверка на выдуманных данных, где верный ответ известен заранее.

    Меряющий инструмент опаснее торгующего: ошибка в выравнивании или в знаке
    не падает, а тихо рисует преимущество. Здесь строится запись, в которой
    сигнал двигает цену с известной силой, и проверяется, что инструмент
    возвращает именно её.

    Первый прогон нарочно сделан почти без шума. Иначе проверка ничего не
    доказывает: на коротком куске и при большом шуме любое число попадает в
    доверительный интервал, и ошибка выравнивания пройдёт незамеченной. При
    малом шуме ответ известен с точностью до десятых — и разойтись с ним
    можно только по-настоящему.

    Ожидаемое валовое: сигнал держится блоком длиной с горизонт, а движение
    приходит СЛЕДОМ, поэтому окно горизонта захватывает движение ровно
    наполовину в среднем. Отсюда 0.5 * сила * E|s| при |s| >= 2, то есть
    примерно 0.5 * 4.0 * 2.37 = 4.7 б.п.
    """
    step, h_ms = 200, 60000
    h = h_ms // step
    n = 1_080_000                     # 60 часов при шаге 200 мс
    spread, fee = 1.0, 0.0
    args = argparse.Namespace(step_ms=step, lag_ms=step, notional=2500)
    ok = True

    cases = ((4.0, 0.05, "сигнал есть, шума почти нет"),
             (0.0, 0.50, "сигнала нет, шум обычный"))
    for edge_bp, noise, name in cases:
        rng = np.random.default_rng(1)
        blocks = n // h + 1
        value = rng.normal(size=blocks)          # сигнал держится блоками
        per_node = np.zeros(n)
        for j in range(1, blocks):
            lo, hi = j * h, min((j + 1) * h, n)
            # движение приходит СЛЕДОМ за сигналом предыдущего блока
            per_node[lo:hi] = edge_bp * value[j - 1] / h
        per_node += rng.normal(scale=noise, size=n)
        g = {"ts": np.arange(n, dtype=np.int64) * step * 1000,
             "seg": np.zeros(n, dtype=np.int32),
             "mid": 100.0 * np.exp(np.cumsum(per_node) / 1e4),
             "spread": np.full(n, spread),
             "delta_raw": np.repeat(value, h)[:n],
             # цена исполнения нарочно НЕсимметричная: покупка дороже продажи.
             # Круг всё равно обязан выйти в spread — и у лонга, и у шорта,
             # иначе перепутаны стороны на входе или на выходе.
             "buy2500": np.full(n, spread * 0.8, dtype=np.float32),
             "sell2500": np.full(n, spread * 0.2, dtype=np.float32)}
        row = next(r for r in measure("ТЕСТ", g, args, fee)
                   if r[0] == h_ms and r[1] == 2.0)
        _, _, n_sig, n_blk, gross, net, t, first, second, ctrl, _ = row
        print(f"\n  {name}: сигналов {n_sig:,}, блоков {n_blk}, "
              f"валовое {gross:.2f}, чистое {net:.2f}, t {t:.2f}, "
              f"контроль {ctrl:.2f}")

        def check(label, condition):
            nonlocal ok
            print(f"    {'ok  ' if condition else 'СБОЙ'}  {label}")
            ok = ok and condition

        check("блоков хватает, чтобы проверка что-то значила", n_blk >= 30)
        check("издержки вычтены ровно один раз",
              abs(net - (gross - (2 * fee + spread))) < 0.05)
        check("контроль равен минус издержкам (случайная сторона не зарабатывает)",
              abs(ctrl + (2 * fee + spread)) < 0.5)
        if edge_bp > 0:
            check("валовое совпало с заложенным 4.7 б.п.", 3.5 < gross < 6.0)
            check("значимо", t > 3)
            check("знак держится на обеих половинах", first > 0 and second > 0)
        else:
            check("без сигнала валовое около нуля", abs(gross) < 1.0)
            check("без сигнала значимости нет", abs(t) < 3)
            check("без сигнала чистое отрицательно", net < 0)

    print("\n  " + ("ВСЕ ПРОВЕРКИ ПРОШЛИ" if ok else "ЕСТЬ СБОИ — верить выводам нельзя"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="all")
    ap.add_argument("--hours", type=float, default=None,
                    help="по умолчанию ВСЯ запись: здесь мощность упирается "
                         "в число наблюдений, и урезать окно нечем платить")
    ap.add_argument("--step-ms", type=int, default=200)
    ap.add_argument("--lag-ms", type=int, default=200)
    ap.add_argument("--notional", type=int, default=2500,
                    choices=FILL_NOTIONALS,
                    help="объём заявки в долларах: издержки считаются как "
                         "настоящая цена исполнения такого объёма по стакану")
    ap.add_argument("--taker-bp", type=float, default=None,
                    help="комиссия тейкера; по умолчанию берётся с биржи "
                         "поинструментно")
    ap.add_argument("--selftest", action="store_true",
                    help="проверить сам инструмент на выдуманных данных")
    a = ap.parse_args()

    if a.selftest:
        print("=" * 100)
        print("  САМОПРОВЕРКА НА ВЫДУМАННЫХ ДАННЫХ")
        print("=" * 100)
        sys.exit(selftest())

    from config import SYMBOLS
    targets = SYMBOLS if a.symbol == "all" else [a.symbol]
    grids = build_grids(targets, a.step_ms, a.hours)

    print("=" * 100)
    print("  СКОЛЬКО ОСТАЁТСЯ ОТ СИГНАЛА ПОСЛЕ ИЗДЕРЖЕК")
    print("  Вход и выход ПО РЫНКУ в сторону дельты сделок за 5 с.")
    print(f"  «Чистое» — за вычетом комиссии и НАСТОЯЩЕЙ цены исполнения")
    print(f"  заявки на ${a.notional:,} по стакану, а не спреда касания.")
    print("=" * 100)

    survivors = {}
    for symbol in targets:
        g = grids.get(symbol)
        if g is None or len(g["mid"]) < 5000:
            print(f"\n  {symbol}: данных мало")
            continue
        fee = a.taker_bp if a.taker_bp is not None else taker_fee_bp(symbol)
        rows = measure(symbol, g, a, fee)
        show(symbol, rows, fee, float(np.median(g["spread"])))
        print(f"  цена круга по рынку:  {depth_note(g)}")
        if f"buy{a.notional}" not in g:
            print("  ВНИМАНИЕ: сетка старая, цены исполнения в ней нет — "
                  "считано по спреду касания")
        elif miss_share(g, a.notional) > 0.05:
            print(f"  ВНИМАНИЕ: в {miss_share(g, a.notional)*100:.0f}% узлов "
                  f"книги не хватало на ${a.notional:,} — они выброшены, "
                  f"и результат смещён в сторону спокойных моментов")
        for h_ms, th, _, _, _, net, t, first, second, ctrl, _ in rows:
            if np.isfinite(net) and net > 0 and first > 0 and second > 0 and t > 3:
                survivors.setdefault((h_ms, th), []).append(symbol)

    print("\n" + "=" * 100)
    print("  ЧТО ПРОШЛО ПРОВЕРКУ")
    print("  Засчитывается только положительное чистое на ОБЕИХ половинах при t > 3.")
    print("=" * 100)
    if not survivors:
        print("  Ни одна пара «горизонт + порог» не прошла ни на одном инструменте.")
    else:
        for (h_ms, th), syms in sorted(survivors.items()):
            label = f"{h_ms/60000:g} мин" if h_ms >= 60000 else f"{h_ms/1000:g} с"
            print(f"  {label:<10}{th:>4.0f}σ   {len(syms)} из {len(targets)}: "
                  + ", ".join(s.split('_')[0] for s in syms))
        print("\n  Одиночный инструмент — это шум: так же выглядел свип по BTC,")
        print("  который развалился на остальных пяти. Смотрим на большинство.")
    print("=" * 100)


if __name__ == "__main__":
    main()
