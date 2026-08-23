#!/usr/bin/env python3
"""Какие пары MEXC вообще годятся для скальпинга.

    python tools/universe.py [--min-turnover 20] [--top 20]

Скальпинг живёт на издержках, а не на идеях: если спред шире цели, стратегии
нет независимо от того, как хорошо ты читаешь стакан. Этот отчёт отбирает
пары по трём числам:

  спред в б.п.   — прямая цена входа и выхода;
  спред / шаг    — сколько шагов цены в спреде. Ровно 1 значит, что улучшить
                   цену нельзя: ты всегда в конце очереди за всей плитой.
                   Больше 1 — можно встать на шаг впереди и пролезть вперёд;
  глубина        — сколько денег стоит у касания и в полосе ±5 б.п.

Обороту в одиночку верить нельзя (его рисуют), поэтому глубина берётся из
живого стакана, а не из статистики биржи.
"""

import argparse
import sys
import time

import requests

DETAIL = "https://contract.mexc.com/api/v1/contract/detail"
TICKER = "https://contract.mexc.com/api/v1/contract/ticker"
DEPTH = "https://contract.mexc.com/api/v1/contract/depth/{}"


def fetch(url, **kw):
    body = requests.get(url, timeout=20, **kw).json()
    if not body.get("success", True):
        sys.exit(f"биржа ответила ошибкой: {body}")
    return body["data"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-turnover", type=float, default=20,
                    help="минимальный оборот за сутки, млн $ (по умолчанию 20)")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    detail = {d["symbol"]: d for d in fetch(DETAIL)}
    tickers = fetch(TICKER)

    rows = []
    for t in tickers:
        s = t["symbol"]
        d = detail.get(s)
        bid, ask = t.get("bid1"), t.get("ask1")
        if not d or not s.endswith("_USDT") or not bid or not ask or ask <= bid:
            continue
        mid = (bid + ask) / 2
        rows.append({
            "symbol": s,
            "spread_bp": (ask - bid) / mid * 1e4,
            "tick_bp": float(d["priceUnit"]) / mid * 1e4,
            "turnover": float(t.get("amount24") or 0),
            "contract": float(d["contractSize"]),
            "mid": mid,
        })

    print(f"бессрочных USDT-пар на MEXC: {len(rows)}")
    edges = [(0.5, "≤ 0.5"), (1, "≤ 1"), (2, "≤ 2"), (5, "≤ 5"), (1e9, "> 5")]
    print(f"\n{'спред, б.п.':<14}{'пар':>6}{'из них оборот > $10 млн':>26}")
    prev = 0
    for hi, label in edges:
        sel = [r for r in rows if prev < r["spread_bp"] <= hi]
        print(f"{label:<14}{len(sel):>6}"
              f"{sum(1 for r in sel if r['turnover'] > 10e6):>26}")
        prev = hi

    cand = [r for r in rows if r["turnover"] > a.min_turnover * 1e6]
    cand.sort(key=lambda r: (r["spread_bp"], -r["turnover"]))
    cand = cand[:a.top]

    print(f"\nГЛУБИНА ПО ЖИВОМУ СТАКАНУ (оборот > ${a.min_turnover:.0f} млн)")
    print(f"{'пара':<15}{'спред б.п.':>11}{'шаг б.п.':>9}{'спред/шаг':>10}"
          f"{'$ касание':>12}{'$ ±5 б.п.':>12}{'оборот $млн':>12}")
    for r in cand:
        try:
            book = fetch(DEPTH.format(r["symbol"]), params={"limit": 200})
        except Exception:
            continue
        if not book.get("bids") or not book.get("asks"):
            continue
        mid = (book["bids"][0][0] + book["asks"][0][0]) / 2
        money = r["contract"] * mid
        touch = (book["bids"][0][1] + book["asks"][0][1]) * money
        band = mid * 0.0005
        near = (sum(v for p, v, *_ in book["bids"] if p >= mid - band) +
                sum(v for p, v, *_ in book["asks"] if p <= mid + band)) * money
        ratio = r["spread_bp"] / r["tick_bp"] if r["tick_bp"] else 0
        print(f"{r['symbol']:<15}{r['spread_bp']:>11.3f}{r['tick_bp']:>9.3f}"
              f"{ratio:>10.1f}{touch/1e3:>11.0f}k{near/1e3:>11.0f}k"
              f"{r['turnover']/1e6:>12.0f}")
        time.sleep(0.15)

    print("\nЧитать так: спред/шаг = 1 — цену улучшить нельзя, ты всегда за всей\n"
          "очередью. Больше 1 — есть куда встать впереди. Шаг сам по себе тоже\n"
          "важен: если один шаг цены больше твоей цели, скальпинг невозможен.")


if __name__ == "__main__":
    main()
