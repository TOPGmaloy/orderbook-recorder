"""Диктофон стакана MEXC: собирает всё и молча пишет на диск.

Что делает:
  * держит один WebSocket на все инструменты, подписан на стакан и ленту;
  * пишет каждое сообщение в parquet как есть, с двумя метками времени;
  * держит книгу в памяти только чтобы заметить потерю пакета и переснять;
  * раз в 5 минут снимает якорный снимок книги — по нему потом проверяется,
    что офлайн-сборка совпадает с реальностью;
  * раз в минуту меряет расхождение часов с биржей, чтобы «задержка» означала
    задержку сети, а не сдвиг локальных часов.

Чего НЕ делает: не торгует, не знает ключей API, не трогает trading-bot.
"""

import asyncio
import json
import logging
import signal
import statistics
import time
from collections import defaultdict, deque

from config import (
    SYMBOLS, SNAPSHOT_SECONDS, CLOCK_SECONDS, STATS_SECONDS,
)
from recorder.book import (
    OrderBook, extract_version, OK, GAP, NOSYNC, SNAPSHOT_OLD, STALE, UNKNOWN,
)
from recorder.mexc_ws import MexcFeed, fetch_snapshot, measure_clock, now_us
from recorder.writer import Writer

log = logging.getLogger("recorder")

RESYNC_COOLDOWN = 2.0        # не чаще одного снимка в 2 секунды на инструмент

# Пауза между ЛЮБЫМИ запросами снимка, общая на процесс. Биржа отвечает кодом
# 510 на пачку одновременных, а у нас их ровно пачка: при старте каждый
# инструмент просит снимок сразу, и якорный цикл раз в две минуты обходит все
# подряд. На шестнадцати инструментах это стоило девяти потерянных снимков из
# шестнадцати — книги собрались только со второй попытки. На плавающей
# вселенной в полторы сотни пар не собрались бы вовсе. Полный обход при 0.3 с
# занимает около минуты, вдвое меньше интервала между якорными снимками.
SNAPSHOT_GAP_S = 0.3


def exchange_ts(data):
    """Время события по часам биржи, в миллисекундах.

    У стакана это `cts` внутри пачки, у ленты — поле `t` первой сделки.
    Верхнеуровневый `ts` сообщения — время отправки, оно грубее; берём его
    только если ничего лучше нет.
    """
    if isinstance(data, dict):
        value = data.get("cts") or data.get("timestamp")
        return int(value) if isinstance(value, (int, float)) else 0
    if isinstance(data, list) and data and isinstance(data[0], dict):
        value = data[0].get("t") or data[0].get("cts")
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


class Recorder:
    def __init__(self):
        self.writer = Writer()
        self.books = {symbol: OrderBook(symbol) for symbol in SYMBOLS}
        self.feed = MexcFeed(SYMBOLS, on_reconnect=self._on_reconnect)
        self.resync_needed = {symbol: asyncio.Event() for symbol in SYMBOLS}
        self.resync_pending = {symbol: False for symbol in SYMBOLS}
        self.counts = defaultdict(int)
        self.lags = defaultdict(lambda: deque(maxlen=3000))
        self.unknown_channels = set()
        self.started = time.time()
        self.stop = asyncio.Event()
        self.snapshot_gate = asyncio.Lock()
        self._snapshot_after = 0.0

    # --- приём сообщений ----------------------------------------------------

    async def handle(self, message, ts_local_us):
        channel = message.get("channel", "")

        if channel == "pong":
            self._record_clock(message.get("data"), ts_local_us)
            return

        if channel == "clientId" or channel.startswith("rs."):
            if channel.startswith("rs.error"):
                log.error("биржа отклонила подписку: %s", message)
            return

        symbol = message.get("symbol") or ""
        data = message.get("data")
        ts_exch = exchange_ts(data) or message.get("ts") or 0
        payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)

        if channel == "push.depth":
            version = extract_version(data)
            self.writer.add(ts_local_us, ts_exch, symbol, "depth", version, payload)
            self.counts["depth"] += 1
            if ts_exch:
                self.lags["depth"].append(ts_local_us // 1000 - ts_exch)

            book = self.books.get(symbol)
            if book is not None:
                status = book.apply_delta(data)
                if status is NOSYNC:
                    await self._request_resync(symbol)
                elif status is GAP:
                    # Разрыв нумерации. Пишем отдельным событием, чтобы при
                    # разборе можно было выбросить испорченный отрезок, а не
                    # считать по нему статистику.
                    had, begin, end = (book.last_gap or (book.version, None, version))
                    self.writer.add(ts_local_us, ts_exch, symbol, "gap", version,
                                    json.dumps({"had_end": had, "got_begin": begin,
                                                "got_end": end}))
                    self.counts["gap"] += 1
                    await self._request_resync(symbol)
                elif status is UNKNOWN:
                    self.counts["unversioned"] += 1
                elif status is STALE:
                    self.counts["stale"] += 1

        elif channel == "push.deal":
            self.writer.add(ts_local_us, ts_exch, symbol, "deal", None, payload)
            self.counts["deal"] += 1
            if ts_exch:
                self.lags["deal"].append(ts_local_us // 1000 - ts_exch)

        elif channel.startswith("push.depth.full"):
            version = extract_version(data)
            self.writer.add(ts_local_us, ts_exch, symbol, "snapshot", version, payload)
            self.counts["snapshot"] += 1

        elif channel not in self.unknown_channels:
            self.unknown_channels.add(channel)
            log.info("незнакомый канал %s — пишу как есть", channel)
            self.writer.add(ts_local_us, ts_exch, symbol, channel, None, payload)

    def _record_clock(self, server_ms, ts_local_us):
        """Сдвиг наших часов относительно биржевых по ответу на ping.

        Схема NTP: серверное время сравнивается с серединой интервала
        «отправили — получили». Ошибка не больше половины RTT, поэтому при
        разборе берётся замер с наименьшим RTT, а не среднее.
        """
        t0 = self.feed.pending_ping_us
        self.feed.pending_ping_us = None
        if not t0 or not isinstance(server_ms, (int, float)):
            return
        rtt_us = ts_local_us - t0
        if rtt_us <= 0:
            return
        offset_ms = (int(server_ms) * 1000 - (t0 + rtt_us // 2)) / 1000
        self.counts["clock"] += 1
        self.writer.add(ts_local_us, int(server_ms), "", "clock", None, json.dumps({
            "source": "ws", "rtt_ms": rtt_us / 1000, "offset_ms": offset_ms,
        }))

    async def _on_reconnect(self):
        """После обрыва книга заведомо неактуальна — переснимаем все."""
        for symbol, book in self.books.items():
            book.reset()
            await self._request_resync(symbol, force=True)

    async def _request_resync(self, symbol, force=False):
        """Просим снимок. Пока прошлый запрос не выполнен, новые не копим."""
        if self.resync_pending.get(symbol) and not force:
            return
        self.resync_pending[symbol] = True
        self.resync_needed[symbol].set()

    # --- фоновые задачи -----------------------------------------------------

    async def resync_worker(self, symbol):
        """Свой рабочий на инструмент: медленный ответ по одному не тормозит остальные."""
        event = self.resync_needed[symbol]
        while not self.stop.is_set():
            await event.wait()
            event.clear()

            # Ждём, пока накопится хотя бы одна пачка. Без неё снимок не с чем
            # стыковать: он приходит отставшим, и первая живая пачка выглядит
            # разрывом. Ожидание — доли секунды, пачки идут раз в ~200 мс.
            book = self.books[symbol]
            for _ in range(40):
                if book.buffer:
                    break
                await asyncio.sleep(0.05)

            data, ts = await self._snapshot(symbol)
            if not data:
                await asyncio.sleep(RESYNC_COOLDOWN)
                event.set()
                continue

            self._write_snapshot(symbol, data, ts)
            status = book.apply_snapshot(data)
            if status is SNAPSHOT_OLD:
                # Снимок не догнал поток — берём следующий, буфер уже копится.
                self.counts["snapshot_old"] += 1
                await asyncio.sleep(0.5)
                event.set()
            else:
                self.resync_pending[symbol] = False

    async def _snapshot(self, symbol):
        """Снимок книги через общий ограничитель частоты.

        Все запросы снимков в процессе идут через это горлышко по одному с
        паузой: пачка одновременных получает от биржи 510 и теряется.
        """
        async with self.snapshot_gate:
            wait = self._snapshot_after - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            data, ts = await asyncio.to_thread(fetch_snapshot, symbol)
            self._snapshot_after = time.monotonic() + SNAPSHOT_GAP_S
            return data, ts

    async def anchor_loop(self):
        """Якорные снимки: не применяем к живой книге, пишем для проверки сборки."""
        while not self.stop.is_set():
            await asyncio.sleep(SNAPSHOT_SECONDS)
            for symbol in SYMBOLS:
                data, ts = await self._snapshot(symbol)
                if data:
                    self._write_snapshot(symbol, data, ts)
                    book = self.books[symbol]
                    if not book.ready and book.apply_snapshot(data) is OK:
                        self.resync_pending[symbol] = False

    async def clock_loop(self):
        """Редкая сверка часов через REST — на случай, если pong не отвечает.

        Основной замер идёт по ping/pong в _record_clock: там дорога та же,
        что у данных, и точность на порядок выше.
        """
        while not self.stop.is_set():
            await asyncio.sleep(CLOCK_SECONDS)
            reading = await asyncio.to_thread(measure_clock)
            if reading:
                reading["source"] = "rest"
                self.writer.add(reading["t1_us"], reading["server_ms"], "", "clock",
                                None, json.dumps(reading))

    def _write_snapshot(self, symbol, data, ts_local_us):
        self.writer.add(ts_local_us, data.get("timestamp") or data.get("cts") or 0,
                        symbol, "snapshot", extract_version(data),
                        json.dumps(data, separators=(",", ":")))
        self.counts["snapshot"] += 1

    async def housekeeping(self):
        """Сброс на диск, уборка и слежение за скачками системных часов.

        Настенные часы могут прыгнуть (поправка NTP, сон машины), монотонные —
        нет. Расхождение между ними означает, что метки времени на этом участке
        врут, и при разборе такой отрезок надо выбрасывать. Пишем это в лог
        событием, а не молчим.
        """
        last_wall, last_mono = time.time(), time.monotonic()
        while not self.stop.is_set():
            self.writer.tick()
            await asyncio.sleep(1)
            wall, mono = time.time(), time.monotonic()
            jump = (wall - last_wall) - (mono - last_mono)
            last_wall, last_mono = wall, mono
            if abs(jump) > 1.0:
                self.counts["clock_jump"] += 1
                log.warning("СКАЧОК ЧАСОВ на %+.1f с — метки времени на этом "
                            "участке ненадёжны", jump)
                self.writer.add(now_us(), 0, "", "status", None,
                                json.dumps({"clock_jump_s": round(jump, 3)}))

    async def stats_loop(self):
        prev = dict(self.counts)
        while not self.stop.is_set():
            await asyncio.sleep(STATS_SECONDS)
            rate = {k: (self.counts[k] - prev.get(k, 0)) / STATS_SECONDS
                    for k in self.counts}
            prev = dict(self.counts)

            parts = [f"аптайм {int(time.time() - self.started)}с",
                     f"стакан {rate.get('depth', 0):.1f}/с",
                     f"сделки {rate.get('deal', 0):.1f}/с"]

            for name in ("depth", "deal"):
                values = list(self.lags[name])
                if values:
                    values.sort()
                    p50 = values[len(values) // 2]
                    p95 = values[int(len(values) * 0.95)]
                    parts.append(f"задержка {name}: медиана {p50}мс, p95 {p95}мс")

            gaps = sum(b.gaps for b in self.books.values())
            parts.append(f"разрывов {gaps}")
            if self.counts.get("clock_jump"):
                parts.append(f"СКАЧКОВ ЧАСОВ {self.counts['clock_jump']}")
            parts.append(f"снимков {self.counts.get('snapshot', 0)}")

            mb, free_gb = self.writer.disk_report()
            parts.append(f"записано {self.writer.written} соб. / {mb:.0f} МБ")
            parts.append(f"свободно {free_gb:.1f} ГБ")
            if self.writer.dropped:
                parts.append(f"ПОТЕРЯНО {self.writer.dropped}")

            line = " | ".join(parts)
            log.info(line)
            self.writer.add(now_us(), 0, "", "status", None, json.dumps({
                "rates": rate, "gaps": gaps, "written": self.writer.written,
                "dropped": self.writer.dropped, "free_gb": free_gb,
                "connects": self.feed.connects, "errors": self.feed.errors,
            }))

    # --- запуск -------------------------------------------------------------

    async def run(self):
        tasks = [
            asyncio.create_task(self.feed.run(self.handle)),
            *[asyncio.create_task(self.resync_worker(s)) for s in SYMBOLS],
            asyncio.create_task(self.anchor_loop()),
            asyncio.create_task(self.clock_loop()),
            asyncio.create_task(self.housekeeping()),
            asyncio.create_task(self.stats_loop()),
        ]
        # Снимки не запрашиваем здесь: это сделает _on_reconnect сразу после
        # подписки. Два источника запроса на старте гонялись между собой —
        # книга успевала собраться и тут же сбрасывалась, давая ложные разрывы.
        await self.stop.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.writer.flush()
        self.writer.close()
        log.info("остановлен, всего записано %d событий", self.writer.written)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    recorder = Recorder()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, recorder.stop.set)
        except NotImplementedError:
            pass
    log.info("инструменты: %s", ", ".join(SYMBOLS))
    await recorder.run()
