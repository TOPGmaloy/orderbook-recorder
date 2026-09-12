"""Запись карты стоимости: один файл на сутки, строки лёгкие.

Диктофон рядом режет файлы по десять минут, потому что parquet получает подпись
при закрытии и открытый файл не читается: при часовой ротации свежий час был бы
невидим для анализа. Здесь обход идёт раз в пять минут, и файл закрывается
после каждого обхода — читать можно всё, кроме последних минут.

Строка весит порядка двадцати чисел, обход даёт около двухсот строк. За сутки
это меньше мегабайта, поэтому хранение здесь длинное: карта спреда по часам
суток имеет смысл на месяцах, а не на двух неделях.
"""

import logging
import shutil
import time
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    COMPRESSION_LEVEL, COSTMAP_DIR, COSTMAP_NOTIONALS, COSTMAP_RETENTION_DAYS,
    MIN_FREE_GB,
)

log = logging.getLogger("costmap")

_FIELDS = [
    ("ts_us", pa.int64()),
    ("symbol", pa.string()),
    ("contract_size", pa.float64()),
    ("turnover_24h", pa.float64()),
    ("bid", pa.float64()),
    ("ask", pa.float64()),
    ("mid", pa.float64()),
    ("spread_pct", pa.float64()),
    ("bid_usd_top", pa.float64()),
    ("ask_usd_top", pa.float64()),
    ("bid_usd_5", pa.float64()),
    ("ask_usd_5", pa.float64()),
    ("levels_bid", pa.int64()),
    ("levels_ask", pa.int64()),
    # Место пары в рейтинге бота на момент замера. None, пока рейтинг не
    # посчитан — первый обход после старта идёт без него.
    ("rank", pa.int64()),
    ("rank_bottom", pa.int64()),
    ("mom_60h", pa.float64()),
]
for _n in COSTMAP_NOTIONALS:
    _FIELDS += [
        (f"buy_{_n}", pa.float64()),
        (f"sell_{_n}", pa.float64()),
        (f"cost_buy_{_n}", pa.float64()),
        (f"cost_sell_{_n}", pa.float64()),
    ]
SCHEMA = pa.schema(_FIELDS)


class CostWriter:
    def __init__(self):
        COSTMAP_DIR.mkdir(parents=True, exist_ok=True)
        self._rows = []
        self._last_retention = 0.0
        self.written = 0
        self.dropped = 0
        self.paused = False

    def add(self, row):
        if self.paused:
            self.dropped += 1
            return
        self._rows.append(row)

    def tick(self):
        """Вызывать после каждого обхода: запись, уборка, охрана диска."""
        self.flush()
        if time.monotonic() - self._last_retention >= 3600:
            self._last_retention = time.monotonic()
            self._retention()
        self._check_disk()

    def flush(self):
        if not self._rows:
            return
        rows, self._rows = self._rows, []
        try:
            table = pa.Table.from_pylist(rows, schema=SCHEMA)
            now = datetime.now(timezone.utc)
            day_dir = COSTMAP_DIR / now.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            path = day_dir / f"cost_{now.strftime('%Y-%m-%dT%H-%M')}.parquet"
            n = 1
            while path.exists():
                path = day_dir / f"cost_{now.strftime('%Y-%m-%dT%H-%M')}.{n}.parquet"
                n += 1
            pq.write_table(table, path, compression="zstd",
                           compression_level=COMPRESSION_LEVEL)
            self.written += len(rows)
        except Exception as exc:
            self.dropped += len(rows)
            log.error("не удалось записать %d строк: %s", len(rows), exc)

    def _retention(self):
        cutoff = datetime.now(timezone.utc).date()
        for day_dir in sorted(COSTMAP_DIR.glob("20*-*-*")):
            try:
                day = datetime.strptime(day_dir.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if (cutoff - day).days > COSTMAP_RETENTION_DAYS:
                shutil.rmtree(day_dir, ignore_errors=True)
                log.info("убран старый день: %s", day_dir.name)

    def _check_disk(self):
        free_gb = shutil.disk_usage(COSTMAP_DIR).free / 1e9
        if free_gb < MIN_FREE_GB and not self.paused:
            self.paused = True
            log.error("свободно %.1f ГБ — запись карты остановлена", free_gb)
        elif free_gb >= MIN_FREE_GB + 1 and self.paused:
            self.paused = False
            log.info("место освободилось (%.1f ГБ) — запись возобновлена", free_gb)
