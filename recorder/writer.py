"""Запись событий на диск.

Формат — parquet со сжатием zstd, каталог на сутки, файл на каждые
ROTATE_MINUTES минут:

    data/2026-08-23/events_2026-08-23T14-30.parquet

Почему не файл на час: подпись parquet ставится при закрытии, открытый файл
прочитать нельзя. Чем реже ротация, тем дольше свежие данные недоступны для
анализа и тем больше теряется при жёстком убийстве процесса.

Пишем СЫРОЙ payload сообщения строкой. Это сознательное решение: если моя
сборка книги где-то ошибается, записи от этого не портятся — разбор делается
офлайн и его можно переделать сколько угодно раз. Потерянные данные вернуть
нельзя, неверный разбор — можно.
"""

import json
import logging
import shutil
import time
from datetime import datetime, timezone, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    DATA_DIR, FLUSH_SECONDS, FLUSH_ROWS, RETENTION_DAYS, MIN_FREE_GB,
    ROTATE_MINUTES, COMPRESSION_LEVEL,
)

log = logging.getLogger("writer")

SCHEMA = pa.schema([
    ("ts_local_us", pa.int64()),   # когда сообщение получили мы, UTC, микросекунды
    ("ts_exch_ms", pa.int64()),    # время биржи из самого сообщения
    ("lag_ms", pa.int64()),        # ts_local - ts_exch, сырая задержка доставки
    ("symbol", pa.string()),
    ("channel", pa.string()),      # depth | deal | snapshot | gap | clock | status
    ("version", pa.int64()),
    ("payload", pa.string()),      # сырое поле data сообщения, как пришло
])


class Writer:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._rows = []
        self._writer = None
        self._bucket = None
        self._last_flush = time.monotonic()
        self._last_retention = 0.0
        self.written = 0
        self.dropped = 0
        self.paused = False

    # --- приём -------------------------------------------------------------

    def add(self, ts_local_us, ts_exch_ms, symbol, channel, version, payload):
        if self.paused:
            self.dropped += 1
            return
        lag = (ts_local_us // 1000 - ts_exch_ms) if ts_exch_ms else 0
        self._rows.append({
            "ts_local_us": int(ts_local_us),
            "ts_exch_ms": int(ts_exch_ms or 0),
            "lag_ms": int(lag),
            "symbol": symbol,
            "channel": channel,
            "version": int(version) if version is not None else -1,
            "payload": payload if isinstance(payload, str) else json.dumps(
                payload, separators=(",", ":"), ensure_ascii=False),
        })
        if len(self._rows) >= FLUSH_ROWS:
            self.flush()

    def tick(self):
        """Вызывать регулярно: сброс по времени, ротация, уборка, охрана диска."""
        if time.monotonic() - self._last_flush >= FLUSH_SECONDS:
            self.flush()
        if time.monotonic() - self._last_retention >= 3600:
            self._last_retention = time.monotonic()
            self._retention()
        self._check_disk()

    # --- запись ------------------------------------------------------------

    def flush(self):
        self._last_flush = time.monotonic()
        if not self._rows:
            return
        rows, self._rows = self._rows, []
        try:
            table = pa.Table.from_pylist(rows, schema=SCHEMA)
            self._rotate()
            self._writer.write_table(table)
            self.written += len(rows)
        except Exception as exc:                      # диск, права, что угодно
            self.dropped += len(rows)
            log.error("не удалось записать %d событий: %s", len(rows), exc)

    def _rotate(self):
        now = datetime.now(timezone.utc)
        bucket = int(now.timestamp()) // (ROTATE_MINUTES * 60)
        if bucket == self._bucket and self._writer is not None:
            return
        self.close()

        start = datetime.fromtimestamp(bucket * ROTATE_MINUTES * 60, timezone.utc)
        day_dir = DATA_DIR / start.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        key = start.strftime("%Y-%m-%dT%H-%M")
        path = day_dir / f"events_{key}.parquet"
        # Дописать в готовый parquet нельзя. Если файл уже есть (перезапуск
        # службы внутри интервала), пишем рядом с суффиксом.
        n = 1
        while path.exists():
            path = day_dir / f"events_{key}.{n}.parquet"
            n += 1
        self._writer = pq.ParquetWriter(path, SCHEMA, compression="zstd",
                                        compression_level=COMPRESSION_LEVEL)
        self._bucket = bucket
        log.info("новый файл: %s", path.name)

    def close(self):
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as exc:
                log.error("ошибка закрытия файла: %s", exc)
            self._writer = None
            self._bucket = None

    # --- диск --------------------------------------------------------------

    def _check_disk(self):
        free_gb = shutil.disk_usage(DATA_DIR).free / 1e9
        if free_gb < MIN_FREE_GB and not self.paused:
            self.paused = True
            log.error("МЕСТО НА ДИСКЕ: свободно %.1f ГБ (< %.1f) — запись остановлена",
                      free_gb, MIN_FREE_GB)
        elif free_gb >= MIN_FREE_GB * 1.5 and self.paused:
            self.paused = False
            log.warning("место освободилось (%.1f ГБ) — запись возобновлена", free_gb)

    def _retention(self):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).date()
        for day_dir in sorted(DATA_DIR.glob("20*-*-*")):
            if not day_dir.is_dir():
                continue
            try:
                day = datetime.strptime(day_dir.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if day < cutoff:
                shutil.rmtree(day_dir, ignore_errors=True)
                log.info("удалён старый каталог: %s", day_dir.name)

    def disk_report(self):
        usage = shutil.disk_usage(DATA_DIR)
        size = sum(p.stat().st_size for p in DATA_DIR.rglob("*.parquet"))
        return size / 1e6, usage.free / 1e9
