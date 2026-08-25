#!/usr/bin/env python3
"""Разбор расхождений сборки: почему уровень есть в снимке, но нет в книге.

    python tools/diagnose.py --symbol BTC_USDT --cases 3 --hours 6

Отчёт говорит, ЧТО книга разошлась со снимком. Здесь мы смотрим, ПОЧЕМУ:
для нескольких первых расхождений печатается вся история проблемной цены
между двумя снимками — была ли она в снимке A, какие пачки её трогали, что
именно они с ней делали и где оказался момент снимка B.

Из этой истории причина обычно видна сразу:

  * цену сняли пачкой, а снимок B её всё ещё показывает → снимок отстаёт от
    собственного номера версии (кэш на стороне биржи), и сравнивать по версии
    нельзя;
  * цену добавила пачка ПОСЛЕ версии B → мы сравниваем не с тем моментом;
  * цену никто не трогал, а объём разъехался → ошибка в применении пачек.
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recorder.book import OrderBook, extract_version
from tools._io import stream


def levels_of(data, side):
    return {float(x[0]): float(x[1]) for x in (data.get(side) or [])}


def run(symbol, cases, hours, window):
    packs = deque(maxlen=window)
    last = None
    found = 0

    for r in stream(symbol=symbol, hours=hours, channels=("depth", "snapshot")):
        data = json.loads(r["payload"])
        if r["channel"] == "depth":
            begin, end = data.get("begin"), data.get("end")
            if isinstance(begin, (int, float)) and isinstance(end, (int, float)):
                packs.append((int(begin), int(end), data, r["ts_local_us"]))
            continue

        version = extract_version(data)
        if version is None:
            continue
        if last is not None and version > last[0]:
            if inspect(symbol, last, (version, data, r["ts_local_us"]), packs):
                found += 1
                if found >= cases:
                    return found
            while packs and packs[0][1] <= last[0]:
                packs.popleft()
        last = (version, data, r["ts_local_us"])
    return found


def inspect(symbol, a, b, packs):
    va, snap_a, ts_a = a
    vb, snap_b, ts_b = b

    book = OrderBook(symbol)
    book.apply_snapshot(snap_a)
    applied, spanning, reached = 0, None, False
    for begin, end, data, ts in packs:
        if end <= va:
            continue
        if begin > book.version + 1:
            return False                      # цепочка рвётся — не наш случай
        if begin > vb:
            reached = True
            break
        if begin <= vb <= end:
            spanning = (begin, end, data)
            reached = True
            break
        book._merge(data)
        book.version = end
        applied += 1
    if not reached:
        return False

    ambiguous = set()
    if spanning:
        ambiguous = set(levels_of(spanning[2], "bids")) | set(levels_of(spanning[2], "asks"))

    for side, sign in (("bids", -1), ("asks", 1)):
        snap = sorted(levels_of(snap_b, side).items(),
                      key=lambda kv: sign * kv[0])[:20]
        own = book.bids if side == "bids" else book.asks
        for price, volume in snap:
            if price in ambiguous or price in own:
                continue
            report(symbol, side, price, volume, va, vb, ts_a, ts_b,
                   snap_a, packs, applied, spanning, book)
            return True
    return False


def report(symbol, side, price, volume, va, vb, ts_a, ts_b,
           snap_a, packs, applied, spanning, book):
    print("=" * 74)
    print(f"  {symbol}  сторона {side}  проблемная цена {price}")
    print(f"  снимок A: версия {va}")
    print(f"  снимок B: версия {vb}   (разница {vb - va:,} изменений, "
          f"{(ts_b - ts_a)/1e6:.0f} с)")
    print(f"  применено пачек: {applied}, книга доведена до версии {book.version}")
    if spanning:
        print(f"  версия B попала внутрь пачки {spanning[0]}..{spanning[1]}")
    else:
        print("  версия B пришлась на стык пачек")
    print("-" * 74)

    in_a = levels_of(snap_a, side).get(price)
    print(f"  в снимке A: {'нет' if in_a is None else f'объём {in_a}'}")
    print(f"  в снимке B: объём {volume}")
    print(f"  в собранной книге: нет")

    print("-" * 74)
    print("  что пачки делали с этой ценой между A и B:")
    touched = 0
    for begin, end, data, ts in packs:
        if end <= va or begin > vb:
            continue
        value = levels_of(data, side).get(price)
        if value is None:
            continue
        touched += 1
        mark = "  <-- пачка со снимком B" if spanning and begin == spanning[0] else ""
        action = "снята" if value == 0 else f"объём {value}"
        print(f"    версии {begin}..{end}: {action}{mark}")
        if touched > 12:
            print("    ...")
            break
    if not touched:
        print("    ни одна пачка её не трогала")

    print("-" * 74)
    after = 0
    for begin, end, data, ts in packs:
        if begin <= vb:
            continue
        if price in levels_of(data, side):
            after += 1
            if after == 1:
                print(f"  ПОСЛЕ версии B её вернула пачка {begin}..{end} "
                      f"(значит снимок B опережает свой номер версии)")
            if after > 3:
                break
    if not after:
        print("  после версии B её никто не возвращал")
    print("=" * 74)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTC_USDT")
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--window", type=int, default=4000,
                    help="сколько последних пачек держать в окне")
    a = ap.parse_args()
    found = run(a.symbol, a.cases, a.hours, a.window)
    if not found:
        print(f"За последние {a.hours:g} ч расхождений по {a.symbol} не нашлось.")


if __name__ == "__main__":
    main()
