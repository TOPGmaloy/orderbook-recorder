"""Карта стоимости входа: весь универс бота, редко, и только про цену исполнения.

Диктофон рядом пишет несколько инструментов глубоко и часто — разрешение,
которое нужно торговле стаканом. Она закрыта. У моментум-бота задача обратная:
он торгует полторы сотни мелких пар, входит по рынку раз в сутки и отдаёт на
исполнении 27% результата. Чтобы этим управлять, нужна не микроструктура одной
монеты, а цена входа по всему универсу, замеренная регулярно.

Что пишется на каждый обход, по строке на пару:

  лучшие цены и спред — то, из чего складывается 91% стоимости;
  глубина у лучшей цены в долларах — сколько влезет, не сдвинув цену;
  настоящая средняя цена исполнения для заявок 150, 500, 2500 и 10000 долларов,
  отдельно на покупку и продажу — проход по книге, а не цена касания.

Последнее и есть смысл файла. Цена касания — это стоимость бесконечно малой
заявки; настоящая заявка идёт вглубь и платит больше, и именно этой разницы
нет ни в одном бэктесте.

Не торгует, ключей не знает, к боту не обращается. Читает публичный REST.
"""

import logging
import time
from datetime import datetime, timezone

import requests

from config import (
    COSTMAP_GAP_S, COSTMAP_LEVELS, COSTMAP_MIN_TURNOVER, COSTMAP_NOTIONALS,
    COSTMAP_RANK_SECONDS, COSTMAP_SECONDS, COSTMAP_UNIVERSE_SECONDS, REST_DEPTH,
    REST_DETAIL, REST_KLINE, REST_TICKER, WATCH_LOOKBACK_HOURS,
)

log = logging.getLogger("costmap")

# Одна сессия на процесс: обход universe — это полторы сотни запросов подряд,
# и без переиспользования соединения каждый платит за рукопожатие TLS. Замер:
# две секунды на пару против сотни миллисекунд, то есть обход не укладывался
# бы в собственный интервал.
_session = requests.Session()
_session.headers["Connection"] = "keep-alive"

# Таймаут кортежем, а не числом: одно значение оставляет запрос висеть, если
# соединение установилось и молчит. Замер на плохом канале — два запроса из
# десяти висели по 500 с при заявленном таймауте в 20, и обход 33 пар занял
# 46 минут вместо 25 секунд. Пять секунд на соединение и пятнадцать на ответ:
# стакан приходит за треть секунды, но список контрактов — это 1192 записи,
# и на плохом канале он выбирает секунды. Всё, что дольше, — сломанная связь,
# а не медленная биржа.
TIMEOUT = (5, 15)


def _get(url, params=None, timeout=TIMEOUT):
    response = _session.get(url, params=params, timeout=timeout)
    body = response.json()
    if not body.get("success", True):
        raise RuntimeError(f"биржа ответила {body.get('code')}")
    return body.get("data")


def universe(min_turnover=None):
    """Пары выше порога оборота с размером контракта.

    Порог задаётся снаружи, потому что у карты стоимости и у следящего состава
    он свой: разойдись они, более строгий молча обрезал бы выборку второго.

    Размер контракта обязателен: объём в стакане приходит в контрактах, и без
    него книга не переводится в доллары. На MEXC контракт равен одной монете
    меньше чем у четверти пар, так что пропустить это нельзя.
    """
    sizes = {}
    for row in _get(REST_DETAIL) or []:
        symbol = row.get("symbol")
        size = row.get("contractSize")
        if symbol and size:
            sizes[symbol] = float(size)

    out = []
    for row in _get(REST_TICKER) or []:
        symbol = row.get("symbol")
        if not symbol or not symbol.endswith("_USDT") or symbol not in sizes:
            continue
        turnover = float(row.get("amount24") or 0)
        if turnover >= (COSTMAP_MIN_TURNOVER if min_turnover is None else min_turnover):
            out.append((symbol, sizes[symbol], turnover))
    out.sort(key=lambda r: -r[2])
    return out


def momentum(symbol, hours):
    """Доходность пары за окно. None, если свечей не хватило."""
    start = int(time.time()) - (hours + 2) * 3600
    data = _get(REST_KLINE.format(symbol=symbol),
                params={"interval": "Min60", "start": start})
    closes = (data or {}).get("close") or []
    if len(closes) < hours:
        return None
    past, now = float(closes[-hours]), float(closes[-1])
    return now / past - 1 if past > 0 else None


def fill(levels, notional, size):
    """Средняя цена исполнения: сумма долларов делится на сумму монет."""
    left, coins, spent = float(notional), 0.0, 0.0
    for level in levels:
        price = float(level[0])
        contracts = float(level[1])
        if price <= 0 or contracts <= 0:
            continue
        available = price * contracts * size
        take = min(left, available)
        coins += take / price
        spent += take
        left -= take
        if left <= 1e-9:
            return spent / coins if coins else None
    return None


def depth_usd(levels, size, limit=1):
    """Сколько долларов стоит у первых `limit` уровней."""
    total = 0.0
    for level in levels[:limit]:
        total += float(level[0]) * float(level[1]) * size
    return total


def probe(symbol, size, turnover):
    """Один снимок книги, посчитанный в доллары."""
    data = _get(REST_DEPTH.format(symbol=symbol), params={"limit": COSTMAP_LEVELS})
    bids = data.get("bids") or []
    asks = data.get("asks") or []
    if not bids or not asks:
        return None
    bid, ask = float(bids[0][0]), float(asks[0][0])
    if bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2

    row = {
        "ts_us": int(time.time() * 1_000_000),
        "symbol": symbol,
        "contract_size": size,
        "turnover_24h": turnover,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread_pct": (ask - bid) / bid * 100,
        "bid_usd_top": depth_usd(bids, size),
        "ask_usd_top": depth_usd(asks, size),
        "bid_usd_5": depth_usd(bids, size, 5),
        "ask_usd_5": depth_usd(asks, size, 5),
        "levels_bid": len(bids),
        "levels_ask": len(asks),
    }
    # Цена круга по объёмам: покупка идёт по ask, продажа по bid. Разница с
    # серединой — то, что бэктест на ценах закрытия не видит вовсе.
    for notional in COSTMAP_NOTIONALS:
        buy = fill(asks, notional, size)
        sell = fill(bids, notional, size)
        row[f"buy_{notional}"] = buy
        row[f"sell_{notional}"] = sell
        row[f"cost_buy_{notional}"] = (buy - mid) / mid * 100 if buy else None
        row[f"cost_sell_{notional}"] = (mid - sell) / mid * 100 if sell else None
    return row


class CostMap:
    """Обходит универс и складывает строки; запись — забота CostWriter."""

    def __init__(self, writer):
        self.writer = writer
        self.pairs = []
        self.universe_at = 0.0
        self.ranks = {}             # символ -> (место сверху, доходность за окно)
        self.ranks_at = 0.0

    def refresh_ranks(self):
        """Место каждой пары в рейтинге бота: 1 — лучший моментум.

        Считается по тем же правилам, что у бота, и нужно, чтобы отчёт умел
        отделять «сколько стоит вход по рынку» от «сколько стоит вход в то, что
        бот действительно берёт».
        """
        scored = []
        # Бюджет времени: рейтинг — довесок к обходу, и если канал лёг, лучше
        # остаться с прошлым рейтингом, чем пропустить сам замер стоимости.
        deadline = time.time() + COSTMAP_SECONDS / 2
        for symbol, _, _ in self.pairs:
            if time.time() > deadline:
                log.warning("рейтинг не досчитан за отведённое время: %d из %d пар",
                            len(scored), len(self.pairs))
                break
            try:
                value = momentum(symbol, WATCH_LOOKBACK_HOURS)
                if value is not None:
                    scored.append((symbol, value))
            except Exception as exc:
                log.debug("%s: рейтинг не посчитался — %s", symbol, exc)
            time.sleep(COSTMAP_GAP_S)
        if len(scored) < 20:
            log.warning("рейтинг посчитан только по %d парам — оставляю прежний",
                        len(scored))
            return
        scored.sort(key=lambda r: -r[1])
        self.ranks = {symbol: (place, value)
                      for place, (symbol, value) in enumerate(scored, 1)}
        self.ranks_at = time.time()
        log.info("рейтинг обновлён по %d парам", len(scored))

    def refresh_universe(self):
        try:
            self.pairs = universe()
            self.universe_at = time.time()
            log.info("универс обновлён: %d пар выше %.0f$ оборота",
                     len(self.pairs), COSTMAP_MIN_TURNOVER)
        except Exception as exc:
            log.warning("универс не обновился: %s", exc)

    def sweep(self):
        """Один полный обход. Возвращает (снято, пропущено, секунд)."""
        if time.time() - self.universe_at > COSTMAP_UNIVERSE_SECONDS:
            self.refresh_universe()
        if not self.pairs:
            return 0, 0, 0.0

        if time.time() - self.ranks_at > COSTMAP_RANK_SECONDS:
            self.refresh_ranks()

        started = time.time()
        # Открытый интерес и ставка по всем парам — одним запросом тикера на
        # обход. MEXC не отдаёт историю OI вообще, только текущее значение, так
        # что ряд для проверки дивергенции OI можно только накопить самим.
        try:
            tickers = {r.get("symbol"): r for r in (_get(REST_TICKER) or [])}
        except Exception as exc:
            log.warning("тикер не прочитан, OI в этом обходе не будет: %s", exc)
            tickers = {}
        # Обход может залипнуть на сломанном канале. Прерываем на двух
        # интервалах: лучше неполный обход вовремя, чем полный когда-нибудь.
        deadline = started + COSTMAP_SECONDS * 2
        done = failed = 0
        total = len(self.ranks)
        for symbol, size, turnover in self.pairs:
            if time.time() > deadline:
                log.warning("обход прерван по времени: не дошли до %d пар",
                            len(self.pairs) - done - failed)
                break
            try:
                row = probe(symbol, size, turnover)
                if row:
                    place, value = self.ranks.get(symbol, (None, None))
                    row["rank"] = place
                    # Место снизу: у шортовой ноги свой край рейтинга, и «дно-2»
                    # удобнее читать как 1 и 2, чем как 166 и 167.
                    row["rank_bottom"] = (total - place + 1) if place else None
                    row["mom_60h"] = value
                    tick = tickers.get(symbol) or {}
                    hold = tick.get("holdVol")
                    last = tick.get("lastPrice")
                    row["oi_contracts"] = float(hold) if hold is not None else None
                    # В долларах: объём в контрактах, умноженный на размер контракта
                    # и цену. Без размера контракта OI разных пар несравним.
                    row["oi_usd"] = (float(hold) * size * float(last)
                                     if hold is not None and last else None)
                    row["funding_rate"] = (float(tick["fundingRate"])
                                           if tick.get("fundingRate") is not None else None)
                    self.writer.add(row)
                    done += 1
                else:
                    failed += 1
            except Exception as exc:
                failed += 1
                log.debug("%s: %s", symbol, exc)
            time.sleep(COSTMAP_GAP_S)
        return done, failed, time.time() - started

    def run(self):
        log.info("карта стоимости запущена: обход раз в %d с, объёмы %s",
                 COSTMAP_SECONDS, ", ".join(f"{n}$" for n in COSTMAP_NOTIONALS))
        self.refresh_universe()
        while True:
            started = time.time()
            done, failed, spent = self.sweep()
            self.writer.tick()
            log.info("обход: снято %d, пропущено %d, за %.0f с", done, failed, spent)
            rest = COSTMAP_SECONDS - (time.time() - started)
            if rest > 0:
                time.sleep(rest)
            elif done:
                log.warning("обход не укладывается в интервал: %.0f с из %d",
                            spent, COSTMAP_SECONDS)
