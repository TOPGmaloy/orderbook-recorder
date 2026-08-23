#!/usr/bin/env python3
"""Отчёт по записанному: сколько, с какой задержкой, целы ли данные.

Запуск:  python tools/report.py [сколько_последних_часов]

Отвечает на три вопроса, без которых считать статистику по стакану нельзя:

  1) какая на самом деле задержка доставки — с поправкой на уход наших часов;
  2) сколько потеряно изменений, то есть какая доля времени непригодна;
  3) сходится ли книга, собранная из инкрементов, со следующим снимком.

Третий пункт — главный. Если из снимка A и всех пачек между ними не получается
снимок B, значит сборка книги где-то врёт, и любые выводы по стакану будут
выводами по выдуманным данным.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa
import pyarrow.parquet as pq

from config import DATA_DIR
from recorder.book import OrderBook, extract_version


def percentile(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def human(ts_us):
    return datetime.fromtimestamp(ts_us / 1e6, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def load(hours=None):
    """Читаем всё, кроме файла, который прямо сейчас пишется.

    У parquet подпись ставится при закрытии, поэтому открытый файл не читается.
    Это нормально, а не поломка: пропускаем его молча-ish и работаем с готовыми.
    """
    files = sorted(DATA_DIR.rglob("events_*.parquet"))
    if not files:
        print(f"В {DATA_DIR} пока пусто.")
        sys.exit(0)
    if hours:
        files = files[-hours:]

    tables, good, writing = [], [], 0
    for path in files:
        try:
            tables.append(pq.read_table(path))
            good.append(path)
        except pa.ArrowInvalid:
            writing += 1
    if not tables:
        print("Готовых файлов пока нет — текущий ещё пишется. "
              "Подождите закрытия интервала и повторите.")
        sys.exit(0)
    if writing:
        print(f"(пропущен файл, который сейчас пишется: {writing})")
    return pa.concat_tables(tables), good


def compare_book(book, snapshot, levels, ambiguous, tol=1e-9):
    """Совпадает ли собранная книга со снимком по верхним уровням.

    `ambiguous` — цены из пачки, внутри которой снимок был сделан. Их сравнивать
    нельзя: снимок мог застать эту пачку применённой наполовину. Всё остальное
    обязано совпасть до последнего лота — расхождение означает ошибку сборки.
    """
    snap = {
        "bids": sorted(((float(p), float(v)) for p, v, *_ in snapshot.get("bids") or []),
                       key=lambda x: -x[0])[:levels],
        "asks": sorted(((float(p), float(v)) for p, v, *_ in snapshot.get("asks") or []),
                       key=lambda x: x[0])[:levels],
    }
    own = {
        "bids": sorted(book.bids.items(), key=lambda x: -x[0])[:levels],
        "asks": sorted(book.asks.items(), key=lambda x: x[0])[:levels],
    }
    for side in ("bids", "asks"):
        own_map = dict(own[side])
        snap_map = dict(snap[side])
        for price, volume in snap[side]:
            if price in ambiguous:
                continue
            if price not in own_map:
                return {"text": f"{side}: в снимке есть {price}, в книге нет",
                        "missing": price}
            if abs(own_map[price] - volume) > tol:
                return {"text": f"{side} {price}: в книге {own_map[price]}, "
                                f"в снимке {volume}", "missing": None}
        if snap[side]:
            edge = min(p for p, _ in snap[side]) if side == "bids" \
                else max(p for p, _ in snap[side])
            for price, volume in own[side]:
                inside = price >= edge if side == "bids" else price <= edge
                if inside and price not in ambiguous and price not in snap_map:
                    return {"text": f"{side}: в книге есть лишний уровень {price}",
                            "missing": None}
    return None


def check_integrity(rows, symbol, levels=20):
    """Снимок A + все пачки между ними == снимок B?

    Главная проверка всего проекта. Тонкость в том, ЧТО с чем сравнивать:
    снимок B сделан биржей в момент, попадающий обычно ВНУТРЬ одной из пачек.
    Наивное «сравним с текущей книгой» даёт расхождение на каждом активном
    инструменте, даже когда сборка идеальна, — и это выглядит как поломка.
    Поэтому книга доводится до пачки, накрывающей версию B, а уровни из этой
    пачки исключаются из сравнения как неопределённые.
    """
    packs, snaps = [], []
    for r in rows:
        if r["symbol"] != symbol:
            continue
        if r["channel"] == "depth":
            data = json.loads(r["payload"])
            begin, end = data.get("begin"), data.get("end")
            if isinstance(begin, (int, float)) and isinstance(end, (int, float)):
                packs.append((int(begin), int(end), data))
        elif r["channel"] == "snapshot":
            data = json.loads(r["payload"])
            version = extract_version(data)
            if version is not None:
                snaps.append((version, data))

    checks = matched = mismatched = incomplete = 0
    holes = bugs = 0
    example = None
    for (va, snap_a), (vb, snap_b) in zip(snaps, snaps[1:]):
        if vb <= va:
            continue
        book = OrderBook(symbol)
        book.apply_snapshot(snap_a)
        ambiguous, reached, chain_ok = set(), False, True
        # Все цены, которые мы вообще могли узнать к моменту сверки. Если
        # недостающий уровень сюда не входит, книга не «сломалась» — этой цены
        # просто никогда не было ни в снимке A (он обрезан по числу уровней),
        # ни в одной пачке. Это дыра покрытия, лечится глубиной снимка.
        seen = {float(level[0]) for side in ("bids", "asks")
                for level in (snap_a.get(side) or [])}

        for begin, end, data in packs:
            if end <= va:
                continue
            if begin > book.version + 1:
                chain_ok = False          # не хватает пачек — обычно после обрыва
                break
            if begin > vb:                # версия B пришлась на стык пачек
                reached = True
                break
            if begin <= vb <= end:        # версия B внутри этой пачки
                ambiguous = {float(level[0])
                             for side in ("bids", "asks")
                             for level in (data.get(side) or [])}
                reached = True
                break
            seen.update(float(level[0]) for side in ("bids", "asks")
                        for level in (data.get(side) or []))
            book._merge(data)
            book.version = end

        if not chain_ok or not reached:
            incomplete += 1
            continue

        checks += 1
        problem = compare_book(book, snap_b, levels, ambiguous)
        if problem:
            mismatched += 1
            if problem["missing"] is not None and problem["missing"] not in seen:
                holes += 1
                problem["text"] += "  ← цены не было ни в снимке A, ни в пачках"
            else:
                bugs += 1
            example = example or problem["text"]
        else:
            matched += 1

    return checks, matched, mismatched, incomplete, example, holes, bugs


def main():
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else None
    table, files = load(hours)
    rows = table.to_pylist()
    rows.sort(key=lambda r: r["ts_local_us"])

    size_mb = sum(f.stat().st_size for f in files) / 1e6
    span_s = (rows[-1]["ts_local_us"] - rows[0]["ts_local_us"]) / 1e6
    symbols = sorted({r["symbol"] for r in rows if r["symbol"]})

    print("=" * 70)
    print(f"  ФАЙЛОВ: {len(files)}   СОБЫТИЙ: {len(rows):,}   РАЗМЕР: {size_mb:.1f} МБ")
    print(f"  ПЕРИОД: {human(rows[0]['ts_local_us'])} .. "
          f"{human(rows[-1]['ts_local_us'])} UTC   ({span_s / 3600:.2f} ч)")
    if span_s > 0:
        print(f"  ПРОГНОЗ ОБЪЁМА: {size_mb / span_s * 86400:.0f} МБ в сутки "
              f"на {len(symbols)} инструмента")
    print("=" * 70)

    by = defaultdict(int)
    for r in rows:
        by[(r["symbol"], r["channel"])] += 1
    print("\nСОБЫТИЯ")
    print(f"  {'инструмент':<12} {'канал':<10} {'всего':>10} {'в секунду':>12}")
    for (symbol, channel), count in sorted(by.items()):
        print(f"  {symbol or '—':<12} {channel:<10} {count:>10,} "
              f"{count / span_s if span_s else 0:>12.2f}")

    # --- часы -------------------------------------------------------------
    clock = [json.loads(r["payload"]) for r in rows if r["channel"] == "clock"]
    clock = [c for c in clock if "rtt_ms" in c]
    ws_clock = [c for c in clock if c.get("source") == "ws"] or clock
    offset_ms, error_ms = 0.0, None
    if ws_clock:
        best = min(ws_clock, key=lambda c: c["rtt_ms"])
        offset_ms = best["offset_ms"]
        error_ms = best["rtt_ms"] / 2
        rtts = [c["rtt_ms"] for c in ws_clock]
        offsets = sorted(c["offset_ms"] for c in ws_clock)
        print("\nЧАСЫ")
        print(f"  замеров по ping/pong: {len(ws_clock)}")
        print(f"  RTT: лучший {min(rtts):.0f} мс, медиана {percentile(rtts, 0.5):.0f} мс")
        print(f"  сдвиг наших часов от биржевых: {offset_ms:+.0f} мс "
              f"(ошибка ≤ {error_ms:.0f} мс)")
        print(f"  разброс оценок сдвига: {offsets[0]:+.0f} .. {offsets[-1]:+.0f} мс")

    print("\nЗАДЕРЖКА ДОСТАВКИ ПОТОКА (метка биржи → наш приём)")
    print(f"  {'канал':<10} {'медиана':>10} {'p95':>10} {'p99':>10}")
    for channel in ("depth", "deal"):
        lags = [r["lag_ms"] + offset_ms for r in rows
                if r["channel"] == channel and r["ts_exch_ms"]]
        if lags:
            print(f"  {channel:<10} {percentile(lags, 0.5):>8.0f} мс "
                  f"{percentile(lags, 0.95):>8.0f} мс {percentile(lags, 0.99):>8.0f} мс")
    if error_ms is not None:
        print(f"  (с поправкой на сдвиг часов {offset_ms:+.0f} мс; "
              f"неопределённость самой поправки ±{error_ms:.0f} мс)")
    print("  Плюс к этому биржа собирает стакан пачками раз в ~200 мс — "
          "их надо прибавить\n  к задержке, чтобы получить отставание нашей "
          "картины от матчинга.")

    # --- целостность -------------------------------------------------------
    jumps = [json.loads(r["payload"]) for r in rows if r["channel"] == "status"]
    jumps = [j for j in jumps if "clock_jump_s" in j]
    if jumps:
        print(f"\n  ВНИМАНИЕ: скачков системных часов {len(jumps)} "
              f"(суммарно {sum(abs(j['clock_jump_s']) for j in jumps):.0f} с) — "
              "метки времени на этих участках недостоверны")

    print("\nНЕПРЕРЫВНОСТЬ ПОТОКА (по сырым записям: begin == прошлый end + 1)")
    for symbol in symbols:
        depth = [r for r in rows if r["channel"] == "depth" and r["symbol"] == symbol]
        breaks = holes = 0
        prev_end = None
        for r in depth:
            data = json.loads(r["payload"])
            begin, end = data.get("begin"), data.get("end")
            if prev_end is not None and isinstance(begin, (int, float)):
                if int(begin) != prev_end + 1:
                    breaks += 1
                    holes += max(0, int(begin) - prev_end - 1)
            if isinstance(end, (int, float)):
                prev_end = int(end)
        share = f"{breaks / len(depth) * 100:.2f}%" if depth else "—"
        print(f"  {symbol:<12} пачек {len(depth):>7,}, обрывов {breaks:>4} ({share:>6}), "
              f"потеряно изменений {holes:>9,}")
    print("  (обрыв после переподключения — нормально; смотрим на долю)")

    print("\nПРОВЕРКА СБОРКИ (снимок A + все пачки == снимок B, топ-20 уровней)")
    for symbol in symbols:
        (checks, matched, mismatched, incomplete, example,
         holes, bugs) = check_integrity(rows, symbol)
        if checks:
            print(f"  {symbol:<12} сверок {checks:>4}, совпало {matched:>4}, "
                  f"разошлось {mismatched:>4} → {matched / checks * 100:5.1f}% годных"
                  + (f", неполных пар {incomplete}" if incomplete else ""))
            if mismatched:
                print(f"  {'':<12} из них дыр покрытия {holes}, "
                      f"настоящих ошибок сборки {bugs}")
            if example:
                print(f"  {'':<12} пример расхождения: {example}")
        else:
            print(f"  {symbol:<12} сверять нечем (нужны два снимка подряд "
                  f"и все пачки между ними; неполных пар: {incomplete})")

    print()


if __name__ == "__main__":
    main()
