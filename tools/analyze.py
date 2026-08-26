#!/usr/bin/env python3
"""Есть ли в стакане MEXC предсказание — проверка до всякого бэктеста.

    python tools/analyze.py --symbol BTC_USDT

Считает по записи признаки микроструктуры и проверяет, предсказывают ли они
будущее движение цены. Пока ответ «нет», писать торговые правила бессмысленно:
никакая интуиция не найдёт того, чего нет в данных.

Как устроена защита от самообмана — это здесь главное:

  * ЗАГЛЯДЫВАНИЕ ВПЕРЁД. Признаки считаются по данным строго до момента t,
    а доходность меряется от t + lag, где lag — задержка решения (по умолчанию
    200 мс: одна пачка биржи). Живой бот не может действовать раньше.
  * ПЕРЕКРЫТИЕ ОКОН. Доходности на горизонте 5 с, снятые каждые 200 мс,
    перекрываются, и обычный t-критерий завышен в разы. Считается поправка
    Ньюи–Уэста по числу перекрывающихся шагов.
  * КОНТРОЛЬ. Те же признаки со случайно переставленными значениями должны
    дать t около нуля. Если и там значимо — ошибка в коде, а не открытие.
  * ИЗДЕРЖКИ. Рядом с каждой оценкой печатается, сколько б.п. нужно отбить,
    чтобы просто выйти в ноль.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from recorder.book import OrderBook
from tools._io import stream, downtime, closed_files

# Горизонты до получаса намеренно. Всё, что мы мерили раньше, кончалось на
# минуте — а при цели в 2-5 б.п. плата за исполнение в 1-2 б.п. съедает от
# трети до всей прибыли, и это безнадёжно по арифметике, а не по сигналу.
# При цели в 30-100 б.п. та же плата составляет проценты. Между минутой и
# двенадцатью часами моментум-бота никто ничего не мерил.
HORIZONS_MS = [1000, 5000, 30000, 60000, 300000, 900000, 1800000]


CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


def _cache_key(symbol, step_ms, hours):
    """Отпечаток данных: имена и размеры файлов. Дописался новый файл —
    отпечаток изменился, кэш пересчитается сам."""
    import hashlib
    files = closed_files(hours, quiet=True)
    mark = "|".join(f"{f.name}:{f.stat().st_size}" for f in files)
    digest = hashlib.sha1(mark.encode()).hexdigest()[:16]
    return CACHE_DIR / f"grid_{symbol}_{step_ms}_{digest}.npz"


def grid(symbol, step_ms, hours=None, use_cache=True):
    """Сетка времени с книгой. Результат кэшируется.

    Пересборка книги по всей записи — самая дорогая операция в проекте:
    распаковка сотен файлов и разбор миллионов сообщений. Для одного прогона
    это минуты, а прогон по шести инструментам читает те же файлы шесть раз.
    Сама сетка при этом крошечная — несколько чисел на узел, десятки мегабайт.
    Поэтому считаем один раз и складываем рядом.
    """
    if use_cache:
        path = _cache_key(symbol, step_ms, hours)
        if path.exists():
            try:
                data = np.load(path)
                return {k: data[k] for k in data.files}
            except Exception:
                path.unlink(missing_ok=True)
        result = _grid_build(symbol, step_ms, hours)
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            for old in CACHE_DIR.glob(f"grid_{symbol}_{step_ms}_*.npz"):
                old.unlink()          # устаревшие отпечатки не копим
            np.savez_compressed(path, **result)
        except Exception:
            pass
        return result
    return _grid_build(symbol, step_ms, hours)


def build_grids(symbols, step_ms, hours=None, use_cache=True):
    """Сетки сразу для нескольких инструментов за ОДИН проход по записи.

    Раньше каждый инструмент читал все файлы заново: восемь инструментов —
    восемь распаковок девятисот мегабайт. Здесь поток читается один раз, а
    книги ведутся параллельно. Уже посчитанное берётся из кэша и в проход не
    попадает.
    """
    need, out = [], {}
    for sym in symbols:
        if use_cache:
            path = _cache_key(sym, step_ms, hours)
            if path.exists():
                try:
                    data = np.load(path)
                    out[sym] = {k: data[k] for k in data.files}
                    continue
                except Exception:
                    path.unlink(missing_ok=True)
        need.append(sym)
    if not need:
        return out

    print(f"  собираю сетки: {', '.join(need)}", flush=True)
    built = _grid_build_many(need, step_ms, hours)
    for sym, result in built.items():
        out[sym] = result
        if use_cache:
            try:
                CACHE_DIR.mkdir(exist_ok=True)
                for old in CACHE_DIR.glob(f"grid_{sym}_{step_ms}_*.npz"):
                    old.unlink()
                np.savez_compressed(_cache_key(sym, step_ms, hours), **result)
            except Exception:
                pass
    return out


class _GridState:
    """Состояние сборки для одного инструмента: книга, узлы сетки, накопители.

    Вынесено в объект ровно затем, чтобы вести восемь книг одновременно в
    одном проходе по файлам. Логика внутри та же, что и при одиночной сборке.
    """

    def __init__(self, symbol, step_ms):
        import heapq
        self.heapq = heapq
        self.symbol = symbol
        self.book = OrderBook(symbol)
        self.step_us = step_ms * 1000
        self.next_ts = None
        self.segment = 0
        self.cols = {k: [] for k in ("ts", "seg", "mid", "spread", "imb1",
                                     "imb5", "imb10", "micro", "ofi_raw",
                                     "delta_raw")}
        self.ofi = self.buy = self.sell = 0.0
        self.prev_bid = self.prev_ask = None

    def _tops(self, side, n, high_first):
        picker = self.heapq.nlargest if high_first else self.heapq.nsmallest
        return sum(side[p] for p in picker(n, side))

    def _emit(self):
        b, a = self.book.best()
        if not b or not a or b[0] >= a[0]:
            return
        m = (b[0] + a[0]) / 2
        c = self.cols
        c["ts"].append(self.next_ts); c["seg"].append(self.segment)
        c["mid"].append(m); c["spread"].append((a[0] - b[0]) / m * 1e4)
        qb, qa = b[1], a[1]
        c["imb1"].append((qb - qa) / (qb + qa) if qb + qa else 0.0)
        for n, key in ((5, "imb5"), (10, "imb10")):
            sb = self._tops(self.book.bids, n, True)
            sa = self._tops(self.book.asks, n, False)
            c[key].append((sb - sa) / (sb + sa) if sb + sa else 0.0)
        c["micro"].append(((b[0] * qa + a[0] * qb) / (qa + qb) - m) / m * 1e4)
        c["ofi_raw"].append(self.ofi)
        c["delta_raw"].append(self.buy - self.sell)
        self.ofi = self.buy = self.sell = 0.0

    def on_hole(self, resume_ts):
        if self.next_ts is not None and self.next_ts >= resume_ts - 1:
            return
        if self.next_ts is not None:
            self.segment += 1
            self.next_ts = resume_ts
            self.book.reset()
            self.prev_bid = self.prev_ask = None

    def on_row(self, r):
        ts = r["ts_local_us"]
        while self.next_ts is not None and ts >= self.next_ts + self.step_us:
            self._emit()
            self.next_ts += self.step_us
        payload = json.loads(r["payload"])
        channel = r["channel"]
        if channel == "snapshot":
            self.book.apply_snapshot(payload)
            if self.next_ts is None and self.book.ready:
                self.next_ts = ts
                self.prev_bid = self.prev_ask = None
        elif channel == "depth":
            if self.book.apply_delta(payload) == "ok":
                b, a = self.book.best()
                if b and a:
                    if self.prev_bid is not None:
                        pb, pa = self.prev_bid, self.prev_ask
                        self.ofi += (b[1] if b[0] > pb[0]
                                     else b[1] - pb[1] if b[0] == pb[0] else -pb[1])
                        self.ofi -= (a[1] if a[0] < pa[0]
                                     else a[1] - pa[1] if a[0] == pa[0] else -pa[1])
                    self.prev_bid, self.prev_ask = b, a
        elif channel == "deal":
            for deal in (payload if isinstance(payload, list) else [payload]):
                try:
                    v = float(deal["v"])
                    if int(deal["T"]) == 1:
                        self.buy += v
                    else:
                        self.sell += v
                except (KeyError, TypeError, ValueError):
                    continue
        elif channel == "gap":
            self.book.ready = False
            self.prev_bid = self.prev_ask = None

    def finish(self):
        c = self.cols
        return {
            "ts": np.array(c["ts"], dtype=np.int64),
            "seg": np.array(c["seg"], dtype=np.int32),
            "mid": np.array(c["mid"]),
            "spread": np.array(c["spread"]),
            "imb1": np.array(c["imb1"]), "imb5": np.array(c["imb5"]),
            "imb10": np.array(c["imb10"]), "micro": np.array(c["micro"]),
            "ofi_raw": np.array(c["ofi_raw"]),
            "delta_raw": np.array(c["delta_raw"]),
        }


def _grid_build_many(symbols, step_ms, hours=None):
    """Один поток, несколько книг. Логика на инструмент та же, что и в
    одиночной сборке, — просто состояние держится в словаре."""
    states = {sym: _GridState(sym, step_ms) for sym in symbols}
    holes = downtime(hours)
    hole_i = 0
    for r in stream(hours=hours, progress=True):
        ts = r["ts_local_us"]
        while hole_i < len(holes) and holes[hole_i][1] <= ts:
            for st in states.values():
                st.on_hole(holes[hole_i][1])
            hole_i += 1
        st = states.get(r["symbol"])
        if st is not None:
            st.on_row(r)
    return {sym: st.finish() for sym, st in states.items()}


def _grid_build(symbol, step_ms, hours=None):
    """Прогон записи: книга на сетке времени + поток сделок и OFI между узлами."""
    book = OrderBook(symbol)
    step_us = step_ms * 1000
    next_ts = None
    segment = 0                 # номер непрерывного куска записи
    holes = downtime(hours)     # когда диктофон реально стоял
    hole_index = 0

    ts, seg, mid, spread = [], [], [], []
    imb1, imb5, imb10 = [], [], []
    micro = []
    ofi_bucket, delta_bucket = [], []
    ofi_acc, buy_acc, sell_acc = 0.0, 0.0, 0.0
    prev_bid = prev_ask = None          # (цена, объём) для расчёта OFI

    import heapq

    def tops(side, n, best_first_high):
        """Сумма объёма на n лучших уровнях. nlargest вместо полной сортировки:
        в книге больше тысячи уровней, а сортировать её на каждом узле сетки
        при часах записи — это минуты впустую."""
        picker = heapq.nlargest if best_first_high else heapq.nsmallest
        return sum(side[p] for p in picker(n, side))

    def emit():
        nonlocal ofi_acc, buy_acc, sell_acc
        b, a = book.best()
        if not b or not a or b[0] >= a[0]:
            return False
        m = (b[0] + a[0]) / 2
        ts.append(next_ts); seg.append(segment); mid.append(m)
        spread.append((a[0] - b[0]) / m * 1e4)
        qb, qa = b[1], a[1]
        imb1.append((qb - qa) / (qb + qa) if qb + qa else 0.0)
        for n, out in ((5, imb5), (10, imb10)):
            sb, sa = tops(book.bids, n, True), tops(book.asks, n, False)
            out.append((sb - sa) / (sb + sa) if sb + sa else 0.0)
        micro.append(((b[0] * qa + a[0] * qb) / (qa + qb) - m) / m * 1e4)
        ofi_bucket.append(ofi_acc)
        delta_bucket.append(buy_acc - sell_acc)
        ofi_acc = 0.0; buy_acc = 0.0; sell_acc = 0.0
        return True

    for r in stream(symbol=symbol, hours=hours):
        # Простой записи: только он рвёт непрерывность. Тишина по инструменту
        # разрывом не считается — книга в это время просто не менялась, и
        # кадры с неизменной ценой это честное наблюдение, а не пропуск.
        while hole_index < len(holes) and next_ts is not None \
                and holes[hole_index][1] <= r["ts_local_us"]:
            if next_ts >= holes[hole_index][0]:
                segment += 1
                next_ts = holes[hole_index][1]
                book.reset()
                prev_bid = prev_ask = None
            hole_index += 1
        while next_ts is not None and r["ts_local_us"] >= next_ts + step_us:
            emit()
            next_ts += step_us
        payload = json.loads(r["payload"])
        if r["channel"] == "snapshot":
            book.apply_snapshot(payload)
            if next_ts is None and book.ready:
                next_ts = r["ts_local_us"]
                prev_bid = prev_ask = None
        elif r["channel"] == "depth":
            if book.apply_delta(payload) == "ok":
                b, a = book.best()
                if b and a:
                    # OFI по Cont-Kukanov-Stoikov: считается по изменениям
                    # лучших котировок, а не по их уровню
                    if prev_bid is not None:
                        if b[0] > prev_bid[0]:
                            ofi_acc += b[1]
                        elif b[0] == prev_bid[0]:
                            ofi_acc += b[1] - prev_bid[1]
                        else:
                            ofi_acc -= prev_bid[1]
                        if a[0] < prev_ask[0]:
                            ofi_acc -= a[1]
                        elif a[0] == prev_ask[0]:
                            ofi_acc -= a[1] - prev_ask[1]
                        else:
                            ofi_acc += prev_ask[1]
                    prev_bid, prev_ask = b, a
        elif r["channel"] == "deal":
            for deal in (payload if isinstance(payload, list) else [payload]):
                try:
                    v = float(deal["v"])
                    if int(deal["T"]) == 1:
                        buy_acc += v
                    else:
                        sell_acc += v
                except (KeyError, TypeError, ValueError):
                    continue
        elif r["channel"] == "gap":
            book.ready = False
            prev_bid = prev_ask = None

    return {
        "ts": np.array(ts, dtype=np.int64),
        "seg": np.array(seg, dtype=np.int32),
        "mid": np.array(mid),
        "spread": np.array(spread),
        "imb1": np.array(imb1), "imb5": np.array(imb5), "imb10": np.array(imb10),
        "micro": np.array(micro),
        "ofi_raw": np.array(ofi_bucket),
        "delta_raw": np.array(delta_bucket),
    }


def rolling(x, k):
    """Скользящая сумма по k узлам, только по прошлому."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[k - 1:] = c[k:] - c[:-k]
    return out


def newey_west_t(x, y, lags):
    """Наклон и t с поправкой на перекрытие окон (Ньюи–Уэст)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 100 or np.std(x) == 0:
        return np.nan, np.nan, n
    x0 = x - x.mean()
    beta = float(np.dot(x0, y - y.mean()) / np.dot(x0, x0))
    resid = (y - y.mean()) - beta * x0
    u = x0 * resid
    s = float(np.dot(u, u))
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1)
        s += 2.0 * w * float(np.dot(u[L:], u[:-L]))
    var = s / (np.dot(x0, x0) ** 2)
    return beta, (beta / np.sqrt(var) if var > 0 else np.nan), n


def cross_check(symbols, step_ms, lag_ms, hours, horizons_ms, features):
    """Один признак на всех инструментах: единственная проверка, которая до
    сих пор убивала все находки. Печатается наклон, t и обе половины записи."""
    print("=" * 96)
    print("  ПРОВЕРКА НА ВСЕХ ИНСТРУМЕНТАХ")
    print("  Находка засчитывается, только если знак совпадает у большинства")
    print("  и держится на обеих половинах записи.")
    print("=" * 96)
    grids = build_grids(symbols, step_ms, hours)
    for h_ms in horizons_ms:
        label = f"{h_ms/60000:g} мин" if h_ms >= 60000 else f"{h_ms/1000:g} с"
        for feat in features:
            print(f"\n  ГОРИЗОНТ {label}   признак: {feat}")
            print(f"    {'инструмент':<13}{'наклон':>9}{'t':>7}{'контроль':>10}"
                  f"{'1-я пол.':>10}{'2-я пол.':>10}{'движение':>11}")
            agree = 0
            for sym in symbols:
                g = grids.get(sym)
                if g is None:
                    continue
                n = len(g["mid"])
                if n < 5000:
                    print(f"    {sym:<13} данных мало")
                    continue
                lag = max(1, lag_ms // step_ms)
                h = max(1, h_ms // step_ms)
                end = n - lag - h
                if end <= 1000:
                    continue
                x = _feature(g, feat, step_ms)
                same = g["seg"][lag + h: lag + h + end] == g["seg"][lag: lag + end]
                fut = np.full(n, np.nan)
                fut[:end] = np.where(
                    same,
                    (g["mid"][lag + h: lag + h + end] / g["mid"][lag: lag + end] - 1) * 1e4,
                    np.nan)
                beta, t, _ = newey_west_t(x, fut, lags=h + lag)
                rng = np.random.default_rng(0)
                xs = x.copy(); good = np.isfinite(xs)
                xs[good] = rng.permutation(xs[good])
                _, t_ctrl, _ = newey_west_t(xs, fut, lags=h + lag)
                half = n // 2
                b1, _, _ = newey_west_t(x[:half], fut[:half], lags=h + lag)
                b2, _, _ = newey_west_t(x[half:], fut[half:], lags=h + lag)
                move = np.nanmedian(np.abs(fut))
                if abs(t) > 3 and np.sign(b1) == np.sign(b2):
                    agree += 1
                print(f"    {sym:<13}{beta:>9.3f}{t:>7.2f}{t_ctrl:>10.2f}"
                      f"{b1:>10.3f}{b2:>10.3f}{move:>11.2f}")
            print(f"    значимо и знак держится на половинах: {agree} из {len(symbols)}")
    print("=" * 96)


def _feature(g, name, step_ms):
    if name.startswith("дельта"):
        window = 5000 if "5" in name else 1000
        v = rolling(g["delta_raw"], max(1, window // step_ms))
    elif name.startswith("OFI"):
        window = 5000 if "5" in name else 1000
        v = rolling(g["ofi_raw"], max(1, window // step_ms))
    else:
        return g.get(name, g["imb1"])
    sd = np.nanstd(v)
    return v / sd if sd > 0 else v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTC_USDT")
    ap.add_argument("--hours", type=float, default=None,
                    help="сколько последних часов брать (по умолчанию всё)")
    ap.add_argument("--step-ms", type=int, default=200,
                    help="шаг сетки; 200 мс — родная частота пачек MEXC")
    ap.add_argument("--lag-ms", type=int, default=200,
                    help="задержка решения: раньше живой бот действовать не может")
    ap.add_argument("--taker-bp", type=float, default=None,
                    help="комиссия taker за одну сторону, б.п.; по умолчанию "
                         "берётся с биржи для этого инструмента")
    a = ap.parse_args()

    if a.symbol == "all":
        from config import SYMBOLS
        cross_check(SYMBOLS, a.step_ms, a.lag_ms, a.hours,
                    [300000, 900000],
                    ["дельта сделок 1с", "дельта сделок 5с"])
        return

    # Ставка своя у каждого инструмента, и половина выборки торгуется без
    # комиссии вовсе. Плоская единица, стоявшая здесь по умолчанию, рисовала
    # порог издержек, которого на этих инструментах нет.
    if a.taker_bp is None:
        from tools.backtest import taker_fee_bp
        a.taker_bp = taker_fee_bp(a.symbol)

    g = grid(a.symbol, a.step_ms, a.hours)
    n = len(g["mid"])
    if n < 300:
        sys.exit(f"Всего {n} узлов сетки — слишком мало, нужно хотя бы несколько часов.")

    span_min = (g["ts"][-1] - g["ts"][0]) / 1e6 / 60
    lag = max(1, a.lag_ms // a.step_ms)

    feats = {
        "imb1  (дисбаланс 1 ур.)": g["imb1"],
        "imb5  (дисбаланс 5 ур.)": g["imb5"],
        "imb10 (дисбаланс 10 ур.)": g["imb10"],
        "micro (сдвиг микроцены)": g["micro"],
        "OFI 1с": rolling(g["ofi_raw"], max(1, 1000 // a.step_ms)),
        "OFI 5с": rolling(g["ofi_raw"], max(1, 5000 // a.step_ms)),
        "дельта сделок 1с": rolling(g["delta_raw"], max(1, 1000 // a.step_ms)),
        "дельта сделок 5с": rolling(g["delta_raw"], max(1, 5000 // a.step_ms)),
    }
    # признаки объёмного типа нормируем, иначе наклон не сравнить между собой
    for k in list(feats):
        v = feats[k]
        s = np.nanstd(v)
        if s > 0 and ("OFI" in k or "дельта" in k):
            feats[k] = v / s

    print("=" * 78)
    parts = int(g["seg"][-1]) + 1
    if parts > 1:
        print(f"  запись состоит из {parts} непрерывных кусков — доходности "
              f"через разрывы не считаются")
    print(f"  {a.symbol}   узлов {n:,}   период {span_min:.0f} мин   "
          f"шаг {a.step_ms} мс   задержка решения {a.lag_ms} мс")
    sp = float(np.median(g["spread"]))
    print(f"  медианный спред {sp:.3f} б.п.")
    print("-" * 78)
    print("  ИЗДЕРЖКИ КРУГА (комиссия обеих сторон + переход через спред)")
    print(f"    taker / taker   {2*a.taker_bp + sp:>6.2f} б.п.")
    print(f"    maker / taker   {a.taker_bp + sp/2:>6.2f} б.п.")
    print(f"    maker / maker   {0.0:>6.2f} б.п.  — комиссия ноль, спред в твою "
          "пользу;\n                            платишь не деньгами, а тем, что "
          "наливают\n                            против тебя")
    print("=" * 78)
    if span_min < 240:
        print("  ВНИМАНИЕ: на таком коротком куске выводы делать нельзя.")
        print("  Это проверка того, что расчёт работает, а не результат.\n")

    mid = g["mid"]
    for h_ms in HORIZONS_MS:
        h = max(1, h_ms // a.step_ms)
        fut = np.full(n, np.nan)
        end = n - lag - h
        if end <= 100:
            continue
        # доходность считается ОТ момента lag после сигнала, а не от самого сигнала
        same = g["seg"][lag + h: lag + h + end] == g["seg"][lag: lag + end]
        raw = (mid[lag + h: lag + h + end] / mid[lag: lag + end] - 1) * 1e4
        fut[:end] = np.where(same, raw, np.nan)
        absmove = np.abs(fut[np.isfinite(fut)])
        med = float(np.median(absmove)); p90 = float(np.percentile(absmove, 90))
        cost_mt = a.taker_bp + sp / 2
        px = float(np.median(mid))
        share = float(np.mean(absmove > cost_mt)) * 100
        label = (f"{h_ms/60000:g} мин" if h_ms >= 60000 else f"{h_ms/1000:g} с")
        print(f"\nГОРИЗОНТ {label} — движение по модулю: медиана "
              f"{med:.3f} б.п. (${med*px/1e4:.2f}), "
              f"в 10% случаев больше {p90:.3f} б.п. (${p90*px/1e4:.2f})")
        print(f"  окон, где движение перекрывает издержки maker/taker "
              f"({cost_mt:.2f} б.п.): {share:.1f}%")
        print(f"  {'признак':<26}{'наклон б.п.':>13}{'t (Ньюи-Уэст)':>16}"
              f"{'t контроля':>13}")
        for name, x in feats.items():
            beta, t, cnt = newey_west_t(x, fut, lags=h + lag)
            rng = np.random.default_rng(0)
            xs = x.copy()
            good = np.isfinite(xs)
            xs[good] = rng.permutation(xs[good])
            _, t_ctrl, _ = newey_west_t(xs, fut, lags=h + lag)
            mark = " *" if abs(t) > 3 else ""
            print(f"  {name:<26}{beta:>13.3f}{t:>16.2f}{t_ctrl:>13.2f}{mark}")

    print("\n" + "=" * 78)
    print("  Читать так: наклон — на сколько б.п. сдвинется цена при отклонении")
    print("  признака на одну единицу (для дисбаланса это весь диапазон от -1 до 1).")
    print("  Значимым считается |t| > 3 при t контроля около нуля. Даже значимый")
    print("  признак бесполезен, если наклон меньше издержек круга.")
    print("=" * 78)


if __name__ == "__main__":
    main()
