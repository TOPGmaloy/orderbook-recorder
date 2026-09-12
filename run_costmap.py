#!/usr/bin/env python3
"""Точка входа карты стоимости. Запуск: python run_costmap.py

Отдельный процесс и отдельная служба: диктофон и карта решают разные задачи,
пишут в разные каталоги и должны падать по отдельности.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recorder.costmap import CostMap
from recorder.cost_writer import CostWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(Path(__file__).resolve().parent / "costmap.log"),
              logging.StreamHandler()],
)

if __name__ == "__main__":
    writer = CostWriter()
    try:
        CostMap(writer).run()
    except KeyboardInterrupt:
        pass
    finally:
        writer.flush()
