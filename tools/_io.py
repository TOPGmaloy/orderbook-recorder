"""Потоковое чтение записи: по одному файлу за раз, а не всё в память.

Зачем: `to_pylist()` раздувает данные примерно в 28 раз против размера на
диске (замерено). Суточная запись на шести инструментах — это 400 МБ файлов и
около 11 ГБ объектов Python, а на сервере 4 ГБ памяти. Любой инструмент,
читающий всё сразу, положил бы машину на первом же серьёзном объёме.

Здесь файлы отдаются по одному и сразу освобождаются. Порядок берётся из
статистики parquet — минимальная метка времени в файле читается из заголовка,
без загрузки данных. Сортировать по имени нельзя: после перезапуска службы
внутри одного интервала появляется файл с суффиксом, и по алфавиту он встаёт
раньше основного.
"""

import sys
import time
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR

TS_COLUMN = "ts_local_us"


def closed_files(hours=None, quiet=False):
    """Готовые файлы в порядке времени. Тот, что сейчас пишется, пропускается."""
    paths = sorted(DATA_DIR.rglob("events_*.parquet"))
    if not paths:
        print(f"В {DATA_DIR} пока пусто.")
        sys.exit(0)

    stamped, writing = [], 0
    for path in paths:
        try:
            meta = pq.ParquetFile(path).metadata
            index = meta.schema.names.index(TS_COLUMN)
            lo = None
            for group in range(meta.num_row_groups):
                stats = meta.row_group(group).column(index).statistics
                if stats is not None and stats.has_min_max:
                    lo = stats.min if lo is None else min(lo, stats.min)
            stamped.append((lo if lo is not None else 0, path))
        except Exception:
            writing += 1          # подпись ещё не дописана — файл в работе
    if not stamped:
        print("Готовых файлов пока нет — текущий ещё пишется. "
              "Подождите закрытия интервала и повторите.")
        sys.exit(0)
    if writing and not quiet:
        print(f"(пропущен файл, который сейчас пишется: {writing})")

    stamped.sort()
    if hours:
        cutoff = stamped[-1][0] - hours * 3600 * 1_000_000
        kept = [item for item in stamped if item[0] >= cutoff]
        stamped = kept or stamped[-1:]
    return [path for _, path in stamped]


def stream(symbol=None, hours=None, channels=None, quiet=False,
           progress=False, columns=None):
    """Строки записи по возрастанию времени, по файлу за раз.

    `progress` печатает, сколько файлов пройдено: на суточной записи разбор
    занимает минуты, и молчащая программа неотличима от зависшей.
    """
    paths = closed_files(hours, quiet=quiet)
    show = progress and len(paths) > 12
    started = time.monotonic()
    for number, path in enumerate(paths, 1):
        if show and (number == 1 or number % 20 == 0 or number == len(paths)):
            print(f"  ... файл {number} из {len(paths)}, "
                  f"{time.monotonic() - started:.0f} с", flush=True)
        try:
            # columns=None читает и payload — самую тяжёлую колонку. Когда
            # нужны только метки времени, это разница в десятки раз.
            table = pq.read_table(path, columns=columns)
            if symbol is not None:
                # отсеиваем инструмент до превращения в объекты Python:
                # при шести инструментах это сразу вшестеро меньше работы
                table = table.filter(pc.equal(table["symbol"], symbol))
            rows = table.to_pylist()
            del table
        except Exception:
            continue
        rows.sort(key=lambda r: r[TS_COLUMN])
        for row in rows:
            if channels is not None and row["channel"] not in channels:
                continue
            yield row
        del rows


class Sampler:
    """Хранит не больше cap значений, прореживая вдвое при переполнении.

    Складывать все задержки за сутки — это миллионы объектов и сотни мегабайт
    ради нескольких процентилей. Систематическое прореживание их не искажает.
    """

    def __init__(self, cap=200_000):
        self.cap = cap
        self.values = []
        self.step = 1
        self.seen = 0

    def add(self, value):
        self.seen += 1
        if self.seen % self.step:
            return
        self.values.append(value)
        if len(self.values) > self.cap:
            self.values = self.values[::2]
            self.step *= 2

    def __len__(self):
        return len(self.values)


def downtime(hours=None, quiet=True):
    """Промежутки, когда диктофон реально не писал.

    Тишина по одному инструменту простоем НЕ является: у золота 2.4
    обновления в секунду, и в спокойные часы оно молчит дольше двух секунд —
    это нормальный рынок, а не пропуск. Раньше такая тишина принималась за
    разрыв, книга сбрасывалась и ждала снимка, и из выборки исчезала треть
    данных.

    Настоящий простой виден по служебным строкам: диктофон пишет их раз в
    минуту независимо от того, идут ли котировки. Нет строк дольше двух с
    половиной минут — значит служба стояла.
    """
    marks = [row[TS_COLUMN] for row in
             stream(channels=("status",), hours=hours, quiet=quiet,
                    columns=[TS_COLUMN, "symbol", "channel"])]
    marks.sort()
    holes = []
    for a, b in zip(marks, marks[1:]):
        if b - a > 150 * 1_000_000:
            holes.append((a, b))
    return holes
