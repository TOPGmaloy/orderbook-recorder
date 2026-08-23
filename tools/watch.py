#!/usr/bin/env python3
"""Живой стакан и лента прямо в терминале.

    python tools/watch.py --symbol BTC_USDT

Отдельное от диктофона подключение к публичному потоку MEXC: ничего не пишет,
ничего не торгует, просто показывает. Работает параллельно с записью и никак
ей не мешает.

Обновление раз в секунду компактным блоком, а не перерисовкой экрана — в
браузерной консоли Hetzner полноэкранные интерфейсы тормозят и мигают.
Выход — Ctrl+C.
"""

import argparse
import asyncio
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from recorder.book import OrderBook
from recorder.mexc_ws import MexcFeed, fetch_snapshot


def contract_info(symbol):
    """Размер контракта и число знаков после запятой из шага цены.

    Округлять по величине цены нельзя: у ETH цена 2472, но шаг 0.01, и при
    одном знаке соседние уровни стакана сливаются в один.
    """
    try:
        body = requests.get("https://contract.mexc.com/api/v1/contract/detail",
                            timeout=15).json()
        for row in body.get("data") or []:
            if row.get("symbol") == symbol:
                text = f"{float(row['priceUnit']):.10f}".rstrip("0")
                digits = len(text.split(".")[1]) if "." in text else 0
                return float(row["contractSize"]), max(0, digits)
    except Exception:
        pass
    fallback = {"BTC_USDT": (0.0001, 1), "ETH_USDT": (0.01, 2), "SOL_USDT": (0.1, 2)}
    return fallback.get(symbol, (1.0, 2))


def money(v):
    if v >= 1e6:
        return f"${v/1e6:.1f}M"
    if v >= 1e3:
        return f"${v/1e3:.0f}k"
    return f"${v:.0f}"


class Watch:
    def __init__(self, symbol, levels, refresh):
        self.symbol = symbol
        self.levels = levels
        self.refresh = refresh
        self.book = OrderBook(symbol)
        self.contract, self.digits = contract_info(symbol)
        self.tape = deque(maxlen=400)      # (ts, price, contracts, side)
        self.last_print = 0.0
        self.msgs = 0

    async def on_message(self, message, ts_us):
        channel = message.get("channel", "")
        if message.get("symbol") != self.symbol:
            return
        data = message.get("data")
        if channel == "push.depth":
            self.msgs += 1
            if self.book.apply_delta(data) == "gap":
                await self.resync()
        elif channel == "push.deal":
            for deal in (data if isinstance(data, list) else [data]):
                try:
                    self.tape.append((time.time(), float(deal["p"]),
                                      float(deal["v"]), int(deal["T"])))
                except (KeyError, TypeError, ValueError):
                    continue

    async def resync(self):
        data, _ = await asyncio.to_thread(fetch_snapshot, self.symbol)
        if data:
            self.book.apply_snapshot(data)

    def render(self):
        b, a = self.book.best()
        if not b or not a:
            return "  собираю книгу..."
        mid = (b[0] + a[0]) / 2
        unit = self.contract * mid
        bids = sorted(self.book.bids, reverse=True)[:self.levels]
        asks = sorted(self.book.asks)[:self.levels]
        peak = max([self.book.bids[p] for p in bids] +
                   [self.book.asks[p] for p in asks] + [1])
        digits = self.digits

        out = []
        for p in reversed(asks):
            v = self.book.asks[p]
            bar = "█" * int(v / peak * 22)
            out.append(f"  {p:>12.{digits}f}  {money(v*unit):>7}  \033[33m{bar}\033[0m")
        spread_bp = (a[0] - b[0]) / mid * 1e4
        qb, qa = b[1], a[1]
        imb = (qb - qa) / (qb + qa) if qb + qa else 0
        out.append(f"  {'':>12}  спред {spread_bp:.3f} б.п.   "
                   f"дисбаланс {imb:+.2f}   {'ПОКУПАТЕЛИ' if imb > 0.2 else 'ПРОДАВЦЫ' if imb < -0.2 else 'ровно'}")
        for p in bids:
            v = self.book.bids[p]
            bar = "█" * int(v / peak * 22)
            out.append(f"  {p:>12.{digits}f}  {money(v*unit):>7}  \033[36m{bar}\033[0m")

        now = time.time()
        recent = [t for t in self.tape if now - t[0] <= 5]
        buy = sum(t[2] for t in recent if t[3] == 1) * unit
        sell = sum(t[2] for t in recent if t[3] == 2) * unit
        out.append("")
        out.append(f"  лента за 5 с: покупали {money(buy)}, продавали {money(sell)}, "
                   f"дельта {'+' if buy >= sell else '−'}{money(abs(buy-sell))}"
                   f"   сделок {len(recent)}")
        last = list(self.tape)[-6:]
        if last:
            marks = "  ".join(
                f"\033[36m{p:.{digits}f}\033[0m" if s == 1 else f"\033[33m{p:.{digits}f}\033[0m"
                for _, p, _, s in last)
            out.append(f"  последние:    {marks}")
        return "\n".join(out)

    async def ticker(self):
        while True:
            await asyncio.sleep(self.refresh)
            stamp = time.strftime("%H:%M:%S")
            print(f"\n─── {self.symbol}  {stamp} ─────────────────────────────")
            print(self.render(), flush=True)

    async def run(self):
        feed = MexcFeed([self.symbol], on_reconnect=self.resync)
        await self.resync()
        asyncio.create_task(self.ticker())
        await feed.run(self.on_message)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTC_USDT")
    ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--refresh", type=float, default=1.0)
    a = ap.parse_args()
    print(f"Живой стакан {a.symbol}. Синий — биды и покупки, жёлтый — аски и "
          f"продажи.\nВыход: Ctrl+C\n")
    try:
        asyncio.run(Watch(a.symbol, a.levels, a.refresh).run())
    except KeyboardInterrupt:
        print("\nостановлено")


if __name__ == "__main__":
    main()
