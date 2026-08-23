"""Подключение к публичному потоку MEXC futures.

Один сокет на все инструменты. Задачи класса: держать соединение живым,
переподписываться после обрыва и отдавать наверх разобранные сообщения.
Никакой логики про стакан здесь нет — только транспорт.
"""

import asyncio
import json
import logging
import ssl
import time

import certifi
import requests
import websockets

from config import (
    WS_URL, REST_DEPTH, REST_TIME, SNAPSHOT_LEVELS,
    PING_SECONDS, RECONNECT_MAX_DELAY,
)

log = logging.getLogger("ws")

# Корневые сертификаты берём из certifi, а не из системы: сборка python.org на
# маке идёт без них, и WebSocket молча падает на проверке сертификата, пока
# requests (у него certifi внутри) работает. На сервере это тоже не мешает.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def now_us():
    return int(time.time() * 1_000_000)


# --- REST -------------------------------------------------------------------

def fetch_snapshot(symbol, timeout=10):
    """Полный снимок книги. Возвращает (data, ts_local_us) или (None, ts)."""
    url = REST_DEPTH.format(symbol=symbol)
    try:
        response = requests.get(url, params={"limit": SNAPSHOT_LEVELS}, timeout=timeout)
        ts = now_us()
        body = response.json()
        if not body.get("success", True):
            log.warning("снимок %s: биржа ответила %s", symbol, body.get("code"))
            return None, ts
        return body.get("data"), ts
    except Exception as exc:
        log.warning("снимок %s не получен: %s", symbol, exc)
        return None, now_us()


def measure_clock(samples=5, timeout=10):
    """Расхождение наших часов с биржевыми.

    Схема как у NTP: засекаем до запроса и после, серверное время сравниваем с
    серединой интервала. Ключевая деталь — из нескольких замеров берём тот, у
    кого RTT минимальный, а не медиану. Ошибка оценки не превышает половины
    RTT, поэтому самый быстрый ответ и есть самый точный; усреднение по
    зашумлённому каналу даёт сдвиг в целую секунду на ровном месте.

    Без этой поправки «задержка доставки» будет мерить не сеть, а уход
    локальных часов.
    """
    best = None
    for _ in range(samples):
        try:
            t0 = now_us()
            response = requests.get(REST_TIME, timeout=timeout)
            t1 = now_us()
            server_ms = response.json().get("data")
            if not isinstance(server_ms, (int, float)):
                continue
            rtt_us = t1 - t0
            if best is None or rtt_us < best["rtt_us"]:
                best = {"t0_us": t0, "t1_us": t1, "server_ms": int(server_ms),
                        "rtt_us": rtt_us,
                        "rtt_ms": rtt_us / 1000,
                        "offset_ms": (int(server_ms) * 1000
                                      - (t0 + rtt_us // 2)) / 1000}
        except Exception as exc:
            log.debug("замер часов не удался: %s", exc)
    if best is None:
        log.warning("не удалось замерить часы ни одной попыткой")
        return None
    best.pop("rtt_us")
    best["samples"] = samples
    return best


# --- WebSocket --------------------------------------------------------------

class MexcFeed:
    """Асинхронный итератор сообщений. Сам переподключается и переподписывается."""

    def __init__(self, symbols, on_reconnect=None):
        self.symbols = symbols
        self.on_reconnect = on_reconnect     # вызывается после каждой переподписки
        self.connects = 0
        self.errors = 0
        self.pending_ping_us = None   # когда отправили ping, ждущий ответа

    def _subscriptions(self):
        for symbol in self.symbols:
            yield {"method": "sub.depth", "param": {"symbol": symbol}}
            yield {"method": "sub.deal", "param": {"symbol": symbol}}

    async def run(self, handler):
        delay = 1
        while True:
            try:
                async with websockets.connect(
                    WS_URL, ssl=SSL_CONTEXT, ping_interval=None,
                    max_queue=None, close_timeout=5,
                ) as socket:
                    self.connects += 1
                    delay = 1
                    log.info("соединение установлено (попытка №%d)", self.connects)

                    for message in self._subscriptions():
                        await socket.send(json.dumps(message))
                    if self.on_reconnect:
                        await self.on_reconnect()

                    self.pending_ping_us = None
                    pinger = asyncio.create_task(self._ping_loop(socket))
                    try:
                        async for raw in socket:
                            ts = now_us()
                            try:
                                message = json.loads(raw)
                            except Exception:
                                continue
                            await handler(message, ts)
                    finally:
                        pinger.cancel()

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.errors += 1
                log.warning("обрыв соединения: %s — повтор через %ds", exc, delay)

            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_DELAY)

    async def _ping_loop(self, socket):
        """Ping не только держит соединение — по нему же меряются часы.

        Ответ `pong` несёт время биржи, а дорога у него та же, что у потока
        данных. Поэтому оценка сдвига часов выходит на порядок точнее, чем
        через REST: RTT вдвое меньше, а ошибка оценки не больше половины RTT.
        """
        while True:
            await asyncio.sleep(PING_SECONDS)
            try:
                self.pending_ping_us = now_us()
                await socket.send(json.dumps({"method": "ping"}))
            except Exception:
                return
