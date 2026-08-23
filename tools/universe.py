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
KLINE = "https://contract.mexc.com/api/v1/contract/kline/{}"


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
    ap.add_argument("--notional", type=float, default=2500,
                    help="размер позиции в $, чтобы оценить долю от касания")
    ap.add_argument("--samples", type=int, default=8,
                    help="сколько раз опросить рынок: спред скачет, "
                         "по одному снимку решать нельзя")
    a = ap.parse_args()

    detail = {d["symbol"]: d for d in fetch(DETAIL)}

    # Спред скачет: у одной и той же пары подряд встречаются 0.12 и 0.85 б.п.
    # По одному снимку решать нельзя — берём медиану по серии опросов.
    print(f"опрашиваю рынок {a.samples} раз, это займёт "
          f"около {a.samples * 3} секунд...")
    series, meta = {}, {}
    for i in range(a.samples):
        for t in fetch(TICKER):
            sym = t["symbol"]
            d = detail.get(sym)
            bid, ask = t.get("bid1"), t.get("ask1")
            if not d or not sym.endswith("_USDT") or not bid or not ask or ask <= bid:
                continue
            mid = (bid + ask) / 2
            series.setdefault(sym, []).append((ask - bid) / mid * 1e4)
            meta[sym] = {"mid": mid, "tick_bp": float(d["priceUnit"]) / mid * 1e4,
                         "turnover": float(t.get("amount24") or 0),
                         "contract": float(d["contractSize"])}
        if i < a.samples - 1:
            time.sleep(3)

    rows = []
    for sym, spreads in series.items():
        if len(spreads) < max(2, a.samples // 2):
            continue
        spreads.sort()
        m = meta[sym]
        rows.append({"symbol": sym, "spread_bp": spreads[len(spreads) // 2],
                     "spread_lo": spreads[0], "spread_hi": spreads[-1],
                     "tick_bp": m["tick_bp"], "turnover": m["turnover"],
                     "contract": m["contract"], "mid": m["mid"]})

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

    now = int(time.time())
    print(f"\nПРИГОДНОСТЬ ДЛЯ СКАЛЬПИНГА (оборот > ${a.min_turnover:.0f} млн, "
          f"размер позиции ${a.notional:,.0f})")
    print(f"{'пара':<15}{'спред':>8}{'разброс':>14}{'сп/шаг':>8}{'ход 1мин':>10}"
          f"{'ход/спред':>11}{'$ касание':>11}{'наш размер':>11}")
    table = []
    for r in cand:
        try:
            book = fetch(DEPTH.format(r["symbol"]), params={"limit": 200})
            kl = fetch(KLINE.format(r["symbol"]),
                       params={"interval": "Min1", "start": now - 86400, "end": now})
        except Exception:
            continue
        if not book.get("bids") or not book.get("asks"):
            continue
        opens, closes = kl.get("open") or [], kl.get("close") or []
        if len(closes) < 200:
            continue
        moves = sorted(abs(float(c) / float(o) - 1) * 1e4
                       for o, c in zip(opens, closes) if float(o))
        move = moves[len(moves) // 2]
        mid = (book["bids"][0][0] + book["asks"][0][0]) / 2
        money = r["contract"] * mid
        touch = (book["bids"][0][1] + book["asks"][0][1]) * money
        share = a.notional / touch * 100 if touch else float("inf")
        table.append((r, move, move / r["spread_bp"] if r["spread_bp"] else 0,
                      touch, share))
        time.sleep(0.12)

    table.sort(key=lambda x: -x[2])
    for r, move, ratio, touch, share in table:
        flag = "" if (ratio >= 20 and share <= 20) else "  ✗"
        spread_range = f"{r['spread_lo']:.2f}–{r['spread_hi']:.2f}"
        print(f"{r['symbol']:<15}{r['spread_bp']:>8.3f}{spread_range:>14}"
              f"{r['spread_bp']/r['tick_bp'] if r['tick_bp'] else 0:>8.1f}"
              f"{move:>10.2f}{ratio:>11.0f}{touch/1e3:>10.0f}k{share:>10.1f}%{flag}")

    print("\nЧто здесь важно, по убыванию:")
    print("  ХОД/СПРЕД — сколько раз медианное минутное движение перекрывает")
    print("     спред. Это и есть плотность возможностей: ниже 20 ловить нечего.")
    print("  НАШ РАЗМЕР — какую долю объёма у лучшей цены займёт позиция. Выше")
    print("     20% значит, что торговать этим на такие деньги нельзя, каким бы")
    print("     красивым ни был ход: ты сам себе весь стакан.")
    print("  СПРЕД/ШАГ = 1 — цену улучшить нельзя, ты всегда в конце очереди;")
    print("     больше 1 — можно встать на шаг впереди.")
    print("  РАЗБРОС спреда за время опроса. Если он широкий, медиана всё равно")
    print("     честнее одного снимка, но к такой паре стоит присмотреться дольше.")
    print("  Знаком ✗ помечены пары, не проходящие первые два условия.")


if __name__ == "__main__":
    main()
