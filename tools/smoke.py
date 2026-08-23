#!/usr/bin/env python3
"""Короткий прогон для проверки, что всё работает.

Запуск:  python tools/smoke.py [секунд]      (по умолчанию 60)

Гоняет диктофон заданное время, потом печатает вердикт. Запускать после
установки и после любой правки — до того, как включать службу насовсем.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SYMBOLS
from recorder.main import Recorder


async def run(seconds):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    recorder = Recorder()
    task = asyncio.create_task(recorder.run())
    await asyncio.sleep(seconds)
    recorder.stop.set()
    await task
    return recorder


def verdict(recorder, seconds):
    counts = recorder.counts
    ok = True
    print("\n" + "=" * 60)
    print(f"  ПРОГОН {seconds} с")
    print("=" * 60)

    def check(label, condition, detail):
        nonlocal ok
        mark = "OK  " if condition else "СБОЙ"
        if not condition:
            ok = False
        print(f"  [{mark}] {label}: {detail}")

    check("соединение", recorder.feed.connects > 0,
          f"подключений {recorder.feed.connects}, обрывов {recorder.feed.errors}")
    check("стакан идёт", counts.get("depth", 0) > 0,
          f"{counts.get('depth', 0)} обновлений ({counts.get('depth', 0) / seconds:.1f}/с)")
    check("лента идёт", counts.get("deal", 0) > 0,
          f"{counts.get('deal', 0)} сделок")
    check("снимки книги", counts.get("snapshot", 0) >= len(SYMBOLS),
          f"{counts.get('snapshot', 0)} шт")
    check("запись на диск", recorder.writer.written > 0,
          f"{recorder.writer.written} событий, потеряно {recorder.writer.dropped}")

    books_ready = sum(1 for b in recorder.books.values() if b.ready)
    check("книга собирается", books_ready > 0,
          f"{books_ready} из {len(recorder.books)} инструментов в рабочем состоянии")

    gaps = sum(b.gaps for b in recorder.books.values())
    unver = sum(b.unversioned for b in recorder.books.values())
    waiting = sum(b.waiting for b in recorder.books.values())
    stale = sum(b.stale for b in recorder.books.values())
    print(f"  [ИНФО] разрывов потока: {gaps}  (пачек в ожидании снимка: "
          f"{waiting}, отброшено как старые: {stale})")
    if counts.get("clock_jump"):
        print(f"  [ВНИМАНИЕ] скачков системных часов: {counts['clock_jump']} — "
              "метки времени на этих участках ненадёжны")
    if unver:
        print(f"  [ИНФО] обновлений без версии: {unver}")
    for symbol, book in recorder.books.items():
        bid, ask = book.best()
        if bid and ask:
            print(f"  [ИНФО] {symbol}: бид {bid[0]} x {bid[1]}  /  "
                  f"аск {ask[0]} x {ask[1]}  (уровней {len(book.bids)}/{len(book.asks)})")

    lags = list(recorder.lags["depth"])
    if lags:
        lags.sort()
        print(f"  [ИНФО] задержка стакана: медиана {lags[len(lags)//2]} мс, "
              f"p95 {lags[int(len(lags)*0.95)]} мс (без поправки на часы)")

    mb, free = recorder.writer.disk_report()
    print(f"  [ИНФО] на диске {mb:.1f} МБ, свободно {free:.1f} ГБ"
          + (f", прогноз {mb / seconds * 86400:.0f} МБ/сутки" if seconds else ""))

    print("=" * 60)
    print("  ВЕРДИКТ: " + ("всё работает" if ok else "есть проблемы, смотри выше"))
    print("=" * 60 + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    secs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rec = asyncio.run(run(secs))
    sys.exit(verdict(rec, secs))
