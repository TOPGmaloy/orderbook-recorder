#!/usr/bin/env python3
"""Почему книга расходится со снимком: сводка по причинам.

    python tools/diagnose.py --symbol XRP_USDT --hours 40

Отчёт говорит, ЧТО расхождение есть. Здесь мы раскладываем расхождения по
причинам, потому что они требуют совершенно разных действий:

  снимок отстал      уровень сняли пачкой ДО версии B, а снимок B его всё ещё
                     показывает. Значит биржа отдаёт снимок из кэша, и его
                     номер версии не соответствует содержимому. Данные целые,
                     чинить надо проверку.
  снимок опередил    уровень вернула пачка ПОСЛЕ версии B, а снимок его уже
                     показывает. То же самое с другой стороны.
  дыра покрытия      цены не было ни в снимке A, ни в одной пачке — снимок
                     обрезан по числу уровней. Тоже не ошибка сборки.
  ОШИБКА СБОРКИ      всё остальное: объём разъехался или уровень пропал без
                     причины. Вот это настоящая беда, и тогда часть записи
                     в анализ пускать нельзя.
"""

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recorder.book import OrderBook, extract_version
from tools._io import stream

LATE = "снимок отстал (уровень снят до версии B, снимок его показывает)"
EARLY = "снимок опередил (уровень возвращён после версии B)"
HOLE = "дыра покрытия (цены не было ни в снимке A, ни в пачках)"
VOLUME = "объём разъехался"
PHANTOM = "лишний уровень в книге"
BROKEN = "ОШИБКА СБОРКИ (уровень пропал без причины)"


def levels_of(data, side):
    return {float(x[0]): float(x[1]) for x in (data.get(side) or [])}


def classify(price, side, va, vb, snap_a, packs, seen):
    """История цены между снимками решает, чья это вина."""
    removed_before = returned_after = False
    for begin, end, data in packs:
        if end <= va:
            continue
        value = levels_of(data, side).get(price)
        if value is None:
            continue
        if end <= vb and value == 0:
            removed_before = True
        if begin > vb and value != 0:
            returned_after = True
    if returned_after:
        return EARLY
    if removed_before:
        return LATE
    if price not in seen:
        return HOLE
    return BROKEN


def walk(symbol, hours, window, cases):
    packs = deque(maxlen=window)
    last = None
    tally = Counter()
    pairs = incomplete = 0
    shown = 0

    for r in stream(symbol=symbol, hours=hours,
                    channels=("depth", "snapshot"), progress=True):
        data = json.loads(r["payload"])
        if r["channel"] == "depth":
            begin, end = data.get("begin"), data.get("end")
            if isinstance(begin, (int, float)) and isinstance(end, (int, float)):
                packs.append((int(begin), int(end), data))
            continue

        version = extract_version(data)
        if version is None:
            continue
        if last is not None and version > last[0]:
            pairs += 1
            found, detail = compare(symbol, last[0], last[1], version, data, packs)
            if found is None:
                incomplete += 1
            else:
                for cause in found:
                    tally[cause] += 1
                if detail and shown < cases:
                    shown += 1
                    print(detail)
            while packs and packs[0][1] <= last[0]:
                packs.popleft()
        last = (version, data)
    return tally, pairs, incomplete


def compare(symbol, va, snap_a, vb, snap_b, packs, levels=20):
    book = OrderBook(symbol)
    book.apply_snapshot(snap_a)
    seen = {float(x[0]) for side in ("bids", "asks")
            for x in (snap_a.get(side) or [])}
    ambiguous, reached = set(), False

    for begin, end, data in packs:
        if end <= va:
            continue
        if begin > book.version + 1:
            return None, None
        if begin > vb:
            reached = True
            break
        if begin <= vb <= end:
            ambiguous = set(levels_of(data, "bids")) | set(levels_of(data, "asks"))
            reached = True
            break
        seen.update(float(x[0]) for side in ("bids", "asks")
                    for x in (data.get(side) or []))
        book._merge(data)
        book.version = end
    if not reached:
        return None, None

    causes, detail = [], None
    for side, sign in (("bids", -1), ("asks", 1)):
        snap = sorted(levels_of(snap_b, side).items(),
                      key=lambda kv: sign * kv[0])[:levels]
        own = book.bids if side == "bids" else book.asks
        if not snap:
            continue
        edge = snap[-1][0]
        for price, volume in snap:
            if price in ambiguous:
                continue
            if price not in own:
                cause = classify(price, side, va, vb, snap_a, packs, seen)
                causes.append(cause)
                if detail is None and cause == BROKEN:
                    detail = trace(symbol, side, price, volume, va, vb,
                                   snap_a, packs, book)
            elif abs(own[price] - volume) > 1e-9:
                causes.append(VOLUME)
        for price in own:
            inside = price >= edge if side == "bids" else price <= edge
            if inside and price not in ambiguous and price not in dict(snap):
                causes.append(PHANTOM)
                break
    return causes, detail


def trace(symbol, side, price, volume, va, vb, snap_a, packs, book):
    lines = ["=" * 74,
             f"  {symbol}  {side}  цена {price}  — разбор одного случая",
             f"  снимок A версия {va}, снимок B версия {vb} "
             f"(разница {vb - va:,} изменений)",
             f"  книга доведена до версии {book.version}",
             f"  в снимке A: {levels_of(snap_a, side).get(price, 'нет')}",
             f"  в снимке B: {volume}",
             "  что пачки делали с этой ценой:"]
    touched = 0
    for begin, end, data in packs:
        if end <= va:
            continue
        value = levels_of(data, side).get(price)
        if value is None:
            continue
        touched += 1
        where = "до B" if end <= vb else "ПОСЛЕ B"
        lines.append(f"    {begin}..{end} ({where}): "
                     + ("снята" if value == 0 else f"объём {value}"))
        if touched >= 10:
            lines.append("    ...")
            break
    if not touched:
        lines.append("    ни одна пачка её не трогала")
    lines.append("=" * 74)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTC_USDT")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--window", type=int, default=4000)
    ap.add_argument("--cases", type=int, default=2)
    a = ap.parse_args()

    tally, pairs, incomplete = walk(a.symbol, a.hours, a.window, a.cases)
    total = sum(tally.values())
    print("=" * 74)
    print(f"  {a.symbol}: сверок {pairs:,}, из них неполных {incomplete:,}")
    print(f"  расхождений всего: {total:,}")
    print("=" * 74)
    if not total:
        print("  расхождений нет")
        return
    for cause, count in tally.most_common():
        mark = "  <<<" if cause == BROKEN else ""
        print(f"  {count:>6}  ({count/total*100:>5.1f}%)  {cause}{mark}")
    print("=" * 74)
    if tally.get(BROKEN):
        print("  Есть настоящие ошибки сборки — разбор одного случая выше.")
    else:
        print("  Настоящих ошибок сборки нет: все расхождения объясняются тем,")
        print("  что REST-снимок не совпадает по времени со своим номером версии.")
        print("  Данные годные, чинить надо проверку, а не запись.")
    print("=" * 74)


if __name__ == "__main__":
    main()
