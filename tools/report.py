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
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools._io import closed_files, stream, Sampler
from recorder.book import OrderBook, extract_version


def percentile(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def human(ts_us):
    return datetime.fromtimestamp(ts_us / 1e6, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def compare_book(book, snapshot, levels, ambiguous, tol=1e-9):
    """Совпадает ли собранная книга со снимком по верхним уровням снимка.

    Сравнение идёт ПО ЦЕНАМ, а не по местам в рейтинге. Разница неочевидна и
    оказалась решающей: раньше проверялось, входит ли цена из снимка в топ-20
    НАШЕЙ книги. Стоит нашей книге содержать на пару уровней больше в том же
    диапазоне — например из-за пачки, внутрь которой попал снимок и которую мы
    поэтому не применяем, — как окно рейтинга съезжает, двадцатая цена снимка
    из него выпадает, и идеально собранная книга объявляется сломанной. Так
    получились 73-96% «годных» при нулевом числе настоящих ошибок.

    `ambiguous` — цены из пачки, внутрь которой попал снимок: она могла быть
    применена наполовину, сравнивать их нельзя.
    """
    for side, sign in (("bids", -1), ("asks", 1)):
        snap = sorted(((float(p), float(v)) for p, v, *_ in snapshot.get(side) or []),
                      key=lambda kv: sign * kv[0])[:levels]
        if not snap:
            continue
        own = book.bids if side == "bids" else book.asks
        snap_map = dict(snap)
        edge = snap[-1][0]

        for price, volume in snap:
            if price in ambiguous:
                continue
            if price not in own:
                return {"text": f"{side}: в снимке есть {price}, в книге нет",
                        "missing": price}
            if abs(own[price] - volume) > tol:
                return {"text": f"{side} {price}: в книге {own[price]}, "
                                f"в снимке {volume}", "missing": None}

        for price in own:
            inside = price >= edge if side == "bids" else price <= edge
            if inside and price not in ambiguous and price not in snap_map:
                return {"text": f"{side}: в книге есть лишний уровень {price}",
                        "missing": None}
    return None


class Integrity:
    """Проверка сборки на потоке: снимок A + пачки между == снимок B.

    Раньше сюда складывались ВСЕ пачки инструмента за весь период — на
    суточной записи это гигабайты. Теперь держится скользящее окно последних
    пачек: между снимками их около шестисот, окна в 2500 хватает с запасом,
    а лишнее выбрасывается сразу после сверки.

    Тонкость прежняя: снимок B сделан биржей в момент, попадающий внутрь одной
    из пачек. Книга доводится до этой пачки, а её уровни исключаются из
    сравнения как неопределённые — иначе расхождение вылезает на каждом
    активном инструменте даже при идеальной сборке.
    """

    def __init__(self, symbol, levels=20):
        self.symbol = symbol
        self.levels = levels
        self.packs = deque(maxlen=2500)
        self.last = None
        self.checks = self.matched = self.mismatched = self.incomplete = 0
        self.holes = self.bugs = 0
        self.example = None

    def on_depth(self, data):
        begin, end = data.get("begin"), data.get("end")
        if isinstance(begin, (int, float)) and isinstance(end, (int, float)):
            self.packs.append((int(begin), int(end), data))

    def on_snapshot(self, data):
        version = extract_version(data)
        if version is None:
            return
        if self.last is not None and version > self.last[0]:
            self._check(self.last[0], self.last[1], version, data)
            while self.packs and self.packs[0][1] <= self.last[0]:
                self.packs.popleft()
        self.last = (version, data)

    def _check(self, va, snap_a, vb, snap_b):
        book = OrderBook(self.symbol)
        book.apply_snapshot(snap_a)
        ambiguous, reached, chain_ok = set(), False, True
        seen = {float(level[0]) for side in ("bids", "asks")
                for level in (snap_a.get(side) or [])}

        for begin, end, data in self.packs:
            if end <= va:
                continue
            if begin > book.version + 1:
                chain_ok = False
                break
            if begin > vb:
                reached = True
                break
            if begin <= vb <= end:
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
            self.incomplete += 1
            return

        self.checks += 1
        problem = compare_book(book, snap_b, self.levels, ambiguous)
        if problem:
            self.mismatched += 1
            if problem["missing"] is not None and problem["missing"] not in seen:
                self.holes += 1
                problem["text"] += "  <- цены не было ни в снимке A, ни в пачках"
            else:
                self.bugs += 1
            self.example = self.example or problem["text"]
        else:
            self.matched += 1



def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else None
    files = closed_files(hours)
    size_mb = sum(f.stat().st_size for f in files) / 1e6

    counts = defaultdict(int)
    lags = {"depth": Sampler(), "deal": Sampler()}
    clock, jumps = [], []
    cont, integ = {}, {}
    first_ts = last_ts = None
    total = 0

    # Один проход по записи: всё, что нужно отчёту, накапливается на лету.
    # Держать строки в памяти нельзя — сутки на шести инструментах это около
    # 11 ГБ объектов Python при 4 ГБ на машине.
    for r in stream(hours=hours, quiet=True, progress=True):
        total += 1
        ts = r["ts_local_us"]
        if first_ts is None:
            first_ts = ts
        last_ts = ts
        symbol, channel = r["symbol"], r["channel"]
        counts[(symbol, channel)] += 1

        if channel in lags:
            if r["ts_exch_ms"]:
                lags[channel].add(r["lag_ms"])
            if channel == "depth" and symbol:
                data = json.loads(r["payload"])
                state = cont.setdefault(symbol, {"packs": 0, "breaks": 0,
                                                 "holes": 0, "prev": None})
                state["packs"] += 1
                begin, end = data.get("begin"), data.get("end")
                if state["prev"] is not None and isinstance(begin, (int, float)):
                    if int(begin) != state["prev"] + 1:
                        state["breaks"] += 1
                        state["holes"] += max(0, int(begin) - state["prev"] - 1)
                if isinstance(end, (int, float)):
                    state["prev"] = int(end)
                integ.setdefault(symbol, Integrity(symbol)).on_depth(data)
        elif channel == "snapshot" and symbol:
            integ.setdefault(symbol, Integrity(symbol)).on_snapshot(
                json.loads(r["payload"]))
        elif channel == "clock":
            reading = json.loads(r["payload"])
            if "rtt_ms" in reading:
                clock.append(reading)
        elif channel == "status":
            note = json.loads(r["payload"])
            if "clock_jump_s" in note:
                jumps.append(note)

    if not total:
        sys.exit("В готовых файлах нет строк.")
    span_s = (last_ts - first_ts) / 1e6
    symbols = sorted({sym for sym, _ in counts if sym})

    print("=" * 70)
    print(f"  ФАЙЛОВ: {len(files)}   СОБЫТИЙ: {total:,}   РАЗМЕР: {size_mb:.1f} МБ")
    print(f"  ПЕРИОД: {human(first_ts)} .. {human(last_ts)} UTC   "
          f"({span_s / 3600:.2f} ч)")
    if span_s > 0:
        print(f"  ПРОГНОЗ ОБЪЁМА: {size_mb / span_s * 86400:.0f} МБ в сутки "
              f"на {len(symbols)} инструмента")
    print("=" * 70)

    print("\nСОБЫТИЯ")
    print(f"  {'инструмент':<12} {'канал':<10} {'всего':>10} {'в секунду':>12}")
    for (symbol, channel), count in sorted(counts.items()):
        print(f"  {symbol or '—':<12} {channel:<10} {count:>10,} "
              f"{count / span_s if span_s else 0:>12.2f}")

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
              f"(ошибка <= {error_ms:.0f} мс)")
        print(f"  разброс оценок сдвига: {offsets[0]:+.0f} .. {offsets[-1]:+.0f} мс")

    print("\nЗАДЕРЖКА ДОСТАВКИ ПОТОКА (метка биржи -> наш приём)")
    print(f"  {'канал':<10} {'медиана':>10} {'p95':>10} {'p99':>10}")
    for channel in ("depth", "deal"):
        values = [v + offset_ms for v in lags[channel].values]
        if values:
            print(f"  {channel:<10} {percentile(values, 0.5):>8.0f} мс "
                  f"{percentile(values, 0.95):>8.0f} мс "
                  f"{percentile(values, 0.99):>8.0f} мс")
    if error_ms is not None:
        print(f"  (с поправкой на сдвиг часов {offset_ms:+.0f} мс; "
              f"неопределённость самой поправки +-{error_ms:.0f} мс)")
    print("  Плюс к этому биржа собирает стакан пачками раз в ~200 мс — "
          "их надо прибавить\n  к задержке, чтобы получить отставание нашей "
          "картины от матчинга.")

    if jumps:
        print(f"\n  ВНИМАНИЕ: скачков системных часов {len(jumps)} "
              f"(суммарно {sum(abs(j['clock_jump_s']) for j in jumps):.0f} с) — "
              "метки времени на этих участках недостоверны")

    print("\nНЕПРЕРЫВНОСТЬ ПОТОКА (по сырым записям: begin == прошлый end + 1)")
    for symbol in symbols:
        state = cont.get(symbol)
        if not state or not state["packs"]:
            continue
        share = f"{state['breaks'] / state['packs'] * 100:.2f}%"
        print(f"  {symbol:<12} пачек {state['packs']:>7,}, "
              f"обрывов {state['breaks']:>4} ({share:>6}), "
              f"потеряно изменений {state['holes']:>9,}")
    print("  (обрыв после переподключения — нормально; смотрим на долю)")

    print("\nПРОВЕРКА СБОРКИ (снимок A + все пачки == снимок B, топ-20 уровней)")
    for symbol in symbols:
        it = integ.get(symbol)
        if not it:
            continue
        if it.checks:
            print(f"  {symbol:<12} сверок {it.checks:>4}, совпало {it.matched:>4}, "
                  f"разошлось {it.mismatched:>4} -> "
                  f"{it.matched / it.checks * 100:5.1f}% годных"
                  + (f", неполных пар {it.incomplete}" if it.incomplete else ""))
            if it.mismatched:
                print(f"  {'':<12} из них дыр покрытия {it.holes}, "
                      f"настоящих ошибок сборки {it.bugs}")
            if it.example:
                print(f"  {'':<12} пример расхождения: {it.example}")
        else:
            print(f"  {symbol:<12} сверять нечем (нужны два снимка подряд; "
                  f"неполных пар: {it.incomplete})")

    print()



if __name__ == "__main__":
    main()
