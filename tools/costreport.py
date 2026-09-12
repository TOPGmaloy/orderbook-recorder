"""Отчёт по карте стоимости входа: во что обходится торговать универсом бота.

Отвечает на четыре вопроса, ради которых карта и пишется:

  1. сколько стоит вход сейчас и насколько это вообще стабильно во времени —
     одна и та же пара за шесть часов меняла стоимость с 0.368% до 0.140%,
     и от этого зависит, чем должен быть порог отказа: свойством пары или
     решением, принимаемым в момент заявки;
  2. в какие часы суток вход дешевле — бот ребалансируется по сетке 14:00 UTC,
     час выбран по Шарпу, и если в этот час спред систематически шире, мы
     платим за выбор фазы дважды;
  3. где ёмкость — при депозите 500$ слот корзины 150$, при 20 тысячах он
     4800$, и карта показывает, на каком размере край съедается;
  4. сколько пар отсекает нынешний порог 0.15% и что остаётся в рейтинге.

Запуск:  cd /root/orderbook-recorder && ./costmap
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import COSTMAP_DIR, COSTMAP_NOTIONALS

# Срез рейтинга, который бот пишет при каждой ребалансировке: что он увидел и
# что из этого взял. Нужен только для сверки — свой рейтинг карта считает сама.
BOT_RANKING_PATH = Path("/root/trading-bot/ranking.json")

TRADING_BOT_THRESHOLD = 0.15        # MAX_ENTRY_COST_PCT бота, в процентах


def load(days=None):
    """Все строки карты. Файлы лёгкие, читаются целиком."""
    rows = []
    for day_dir in sorted(COSTMAP_DIR.glob("20*-*-*")):
        if days and day_dir.name < days:
            continue
        for path in sorted(day_dir.glob("cost_*.parquet")):
            try:
                rows.extend(pq.read_table(path).to_pylist())
            except Exception as exc:
                print(f"  пропущен {path.name}: {exc}")
    return rows


def median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def quantile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(int(q * len(values)), len(values) - 1)]


def main():
    rows = load()
    if not rows:
        print(f"В {COSTMAP_DIR} пусто. Служба ещё не писала или не запущена:")
        print("  systemctl status orderbook-costmap")
        return

    stamps = [r["ts_us"] for r in rows]
    first = datetime.fromtimestamp(min(stamps) / 1e6, timezone.utc)
    last = datetime.fromtimestamp(max(stamps) / 1e6, timezone.utc)
    pairs = {r["symbol"] for r in rows}
    sweeps = len({r["ts_us"] // 60_000_000 for r in rows})
    print(f"{'=' * 92}\nКАРТА СТОИМОСТИ ВХОДА\n{'=' * 92}")
    print(f"  строк {len(rows):,}, пар {len(pairs)}, обходов примерно {sweeps}")
    print(f"  период {first:%Y-%m-%d %H:%M} .. {last:%Y-%m-%d %H:%M} UTC "
          f"({(last - first).total_seconds() / 3600:.1f} ч)")

    slot = COSTMAP_NOTIONALS[0]
    key = f"cost_buy_{slot}"

    # 1. Распределение и устойчивость
    print(f"\n{'=' * 92}\n1. СКОЛЬКО СТОИТ ВХОД НА СЛОТ {slot}$ (покупка против середины)\n{'=' * 92}")
    costs = [r[key] for r in rows if r.get(key) is not None]
    print(f"  медиана {median(costs):.4f}%, p75 {quantile(costs, 0.75):.4f}%, "
          f"p90 {quantile(costs, 0.90):.4f}%, p99 {quantile(costs, 0.99):.4f}%, "
          f"максимум {max(costs):.4f}%")
    over = sum(1 for c in costs if c > TRADING_BOT_THRESHOLD)
    print(f"  выше порога бота {TRADING_BOT_THRESHOLD}%: {over} замеров из "
          f"{len(costs)} ({over / len(costs) * 100:.1f}%)")

    by_pair = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            by_pair[r["symbol"]].append(r[key])
    swing = [(s, median(v), max(v) - min(v), len(v))
             for s, v in by_pair.items() if len(v) >= 5]
    swing.sort(key=lambda x: -x[2])
    print("\n  устойчивость: пары с самым гуляющим входом")
    print(f"  {'пара':<16}{'замеров':>9}{'медиана':>10}{'размах':>10}")
    for symbol, med, span, n in swing[:8]:
        print(f"  {symbol:<16}{n:>9}{med:>9.3f}%{span:>9.3f}%")
    stable = sum(1 for _, _, span, _ in swing if span < TRADING_BOT_THRESHOLD)
    print(f"  у {stable} пар из {len(swing)} размах меньше самого порога — "
          f"для них дороговизна свойство пары, а не момента")

    # 2. Карта по часам
    print(f"\n{'=' * 92}\n2. ПО ЧАСАМ СУТОК (UTC): когда вход дешевле\n{'=' * 92}")
    by_hour = defaultdict(list)
    spread_hour = defaultdict(list)
    for r in rows:
        hour = datetime.fromtimestamp(r["ts_us"] / 1e6, timezone.utc).hour
        if r.get(key) is not None:
            by_hour[hour].append(r[key])
        spread_hour[hour].append(r["spread_pct"])
    if by_hour:
        print(f"  {'час':<6}{'медиана входа':>16}{'медиана спреда':>17}{'замеров':>10}")
        best = min(by_hour, key=lambda h: median(by_hour[h]))
        for hour in sorted(by_hour):
            mark = '  <- дешевле всего' if hour == best else ''
            mark += '  <- сетка бота' if hour == 14 else ''
            print(f"  {hour:02d}:00{median(by_hour[hour]):>15.4f}%"
                  f"{median(spread_hour[hour]):>16.4f}%{len(by_hour[hour]):>10}{mark}")
        if 14 in by_hour:
            gap = median(by_hour[14]) - median(by_hour[best])
            print(f"\n  разница между сеткой 14:00 и самым дешёвым часом: "
                  f"{gap:+.4f} п.п. на ногу")
            print(f"  на четырёх ногах круга это {gap * 8:+.3f}% депозита за цикл "
                  f"при полном номинале")

    # 3. Ёмкость
    print(f"\n{'=' * 92}\n3. ЁМКОСТЬ: как растёт стоимость с размером заявки\n{'=' * 92}")
    print(f"  {'заявка':<12}{'медиана':>11}{'p90':>10}{'книга не тянет':>17}")
    for notional in COSTMAP_NOTIONALS:
        k = f"cost_buy_{notional}"
        got = [r[k] for r in rows if r.get(k) is not None]
        missing = sum(1 for r in rows if r.get(k) is None)
        if not got:
            continue
        print(f"  {str(notional) + '$':<12}{median(got):>10.4f}%{quantile(got, 0.9):>9.4f}%"
              f"{missing / len(rows) * 100:>16.1f}%")
    print("\n  «книга не тянет» — доля замеров, где видимой глубины не хватило на заявку.")
    print("  Это и есть потолок: дальше растёт не цена, а невозможность войти вовсе.")

    # 4. Что берёт бот
    print(f"\n{'=' * 92}\n4. ЧТО СТОИТ ТО, ЧТО БОТ ДЕЙСТВИТЕЛЬНО БЕРЁТ\n{'=' * 92}")
    ranked = [r for r in rows if r.get("rank") and r.get(key) is not None]
    if not ranked:
        print("  рейтинг ещё не посчитан — первый расчёт идёт через час после старта")
    else:
        groups = [
            ("верх рейтинга, места 1-2 (лонги бота)", lambda r: r["rank"] <= 2),
            ("места 3-6", lambda r: 3 <= r["rank"] <= 6),
            ("середина", lambda r: r["rank"] > 6 and r["rank_bottom"] > 6),
            ("места 3-6 снизу", lambda r: 3 <= r["rank_bottom"] <= 6),
            ("низ рейтинга, места 1-2 (шорты бота)", lambda r: r["rank_bottom"] <= 2),
        ]
        print(f"  {'группа':<40}{'замеров':>9}{'медиана':>11}{'p90':>10}{'выше порога':>13}")
        for label, test in groups:
            part = [r[key] for r in ranked if test(r)]
            if not part:
                continue
            over = sum(1 for c in part if c > TRADING_BOT_THRESHOLD)
            print(f"  {label:<40}{len(part):>9}{median(part):>10.4f}%"
                  f"{quantile(part, 0.9):>9.4f}%{over / len(part) * 100:>12.0f}%")
        edges = [r[key] for r in ranked if r["rank"] <= 2 or r["rank_bottom"] <= 2]
        middle = [r[key] for r in ranked if r["rank"] > 6 and r["rank_bottom"] > 6]
        if edges and middle:
            print(f"\n  края рейтинга против середины: {median(edges):.4f}% против "
                  f"{median(middle):.4f}% — "
                  f"{'дороже' if median(edges) > median(middle) else 'дешевле'} в "
                  f"{max(median(edges), median(middle)) / max(min(median(edges), median(middle)), 1e-9):.1f} раза")
            print("  Моментум лезет именно в края: туда только что набежали, и спред там свой.")

    # 5. Сверка с тем, что бот записал сам
    print(f"\n{'=' * 92}\n5. СВЕРКА РЕЙТИНГА С БОТОМ\n{'=' * 92}")
    try:
        snap = json.loads(BOT_RANKING_PATH.read_text())
    except Exception as exc:
        print(f"  срез рейтинга бота не прочитан ({exc}) — сверка пропущена")
    else:
        theirs = {row["symbol"].split("/")[0] + "_USDT": row["rank"]
                  for row in snap.get("rows", [])}
        latest = {}
        for r in ranked:
            if r["symbol"] in theirs:
                latest.setdefault(r["symbol"], r)
                if r["ts_us"] > latest[r["symbol"]]["ts_us"]:
                    latest[r["symbol"]] = r
        print(f"  срез бота от {snap.get('at')}, универс {snap.get('universe')} пар")
        print(f"  {'пара':<18}{'у бота':>9}{'у карты':>10}{'вход':>10}")
        for symbol, their_rank in sorted(theirs.items(), key=lambda kv: kv[1]):
            mine = latest.get(symbol)
            ours = mine["rank"] if mine else None
            cost = f"{mine[key]:.4f}%" if mine and mine.get(key) is not None else "—"
            print(f"  {symbol:<18}{their_rank:>9}{(ours if ours else '—'):>10}{cost:>10}")
        both = [(theirs[s], latest[s]["rank"]) for s in theirs if s in latest]
        if both:
            close = sum(1 for a, b in both if abs(a - b) <= 3)
            print(f"\n  совпало по месту в пределах трёх позиций: {close} из {len(both)}")
            print("  Расхождение ожидаемо: срез бота снят в момент ребалансировки,")
            print("  карта считает рейтинг раз в час и по своему списку пар.")

    # 6. Что отсекает порог
    print(f"\n{'=' * 92}\n6. ЧТО ОТСЕКАЕТ ПОРОГ {TRADING_BOT_THRESHOLD}%\n{'=' * 92}")
    always = [s for s, v in by_pair.items()
              if len(v) >= 5 and min(v) > TRADING_BOT_THRESHOLD]
    never = [s for s, v in by_pair.items()
             if len(v) >= 5 and max(v) <= TRADING_BOT_THRESHOLD]
    sometimes = [s for s, v in by_pair.items()
                 if len(v) >= 5 and min(v) <= TRADING_BOT_THRESHOLD < max(v)]
    print(f"  никогда не проходят : {len(always)} пар")
    print(f"  проходят всегда     : {len(never)} пар")
    print(f"  то да, то нет       : {len(sometimes)} пар")
    if always:
        print("\n  дорогие всегда: " + ", ".join(sorted(always)[:15])
              + (" ..." if len(always) > 15 else ""))
    if sometimes:
        print("  пограничные: " + ", ".join(sorted(sometimes)[:15])
              + (" ..." if len(sometimes) > 15 else ""))


if __name__ == "__main__":
    main()
