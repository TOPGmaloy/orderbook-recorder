"""Самопроверка состава записи: кандидаты, портфель, хвост.

Логика набора решает, что вообще попадёт на диск, и ошибка здесь не падает, а
тихо пишет не те пары. Проверяется на выдуманных данных, без биржи и без бота:
подменяются только два источника — рейтинг и файл состояния.

Запуск:  cd /root/orderbook-recorder && ./watchtest
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recorder import watchlist as wl


def check(condition, message):
    if not condition:
        print(f"  ПРОВАЛ: {message}")
        sys.exit(1)
    print(f"  ок: {message}")


def main():
    print("1. Перевод символа из формата бота в формат записи")
    check(wl.to_recorder("NIULAI/USDT:USDT") == "NIULAI_USDT", "пара с суффиксом")
    check(wl.to_recorder("HNT/USDT") == "HNT_USDT", "пара без суффикса")

    # Дальше источники подменяются: биржа и бот в самопроверке не участвуют.
    real_portfolio = wl.portfolio
    ranking = {"A_USDT", "B_USDT", "Y_USDT", "Z_USDT"}
    held = set()
    wl.candidates = lambda: set(ranking)
    wl.portfolio = lambda: set(held)

    print("\n2. Первая сборка: кандидаты плюс портфель")
    watch = wl.Watchlist(["FALLBACK_USDT"])
    held = {"A_USDT"}
    add, drop = watch.refresh()
    check(watch.current == ranking, "набор равен рейтингу, портфель уже внутри")
    check("FALLBACK_USDT" in drop, "запасной список снят, как только пришли данные")

    print("\n3. Бот взял пару вне рейтинга — она добавляется")
    held = {"A_USDT", "NEW_USDT"}
    add, drop = watch.refresh()
    check("NEW_USDT" in add and "NEW_USDT" in watch.current, "взятая пара в наборе")

    print("\n4. Пара выпала из портфеля — держится хвостом, а не снимается сразу")
    held = {"A_USDT"}
    add, drop = watch.refresh()
    check("NEW_USDT" not in drop, "выпавшая пара не снята немедленно")
    check("NEW_USDT" in watch.leaving, "выпавшая пара поставлена в хвост")
    check("NEW_USDT" in watch.current, "и продолжает писаться")

    print("\n5. Хвост истёк — пара снимается")
    watch.leaving["NEW_USDT"] = time.time() - 1
    add, drop = watch.refresh()
    check("NEW_USDT" in drop, "после хвоста подписка снята")
    check("NEW_USDT" not in watch.current, "и пары в наборе больше нет")

    print("\n6. Пара вернулась в портфель, пока была в хвосте")
    held = {"A_USDT", "B_USDT"}
    watch.leaving["B_USDT"] = time.time() + 600
    watch.refresh()
    check("B_USDT" not in watch.leaving, "вернувшаяся пара убрана из хвоста")

    print("\n7. Источники недоступны — набор не рушится")
    wl.candidates = lambda: None
    wl.portfolio = lambda: None
    before = set(watch.current)
    add, drop = watch.refresh()
    check(watch.current == before, "при отказе обоих источников набор прежний")
    check(not add and not drop, "и подписки не трогаются")

    print("\n8. Пустые источники не оставляют запись без единой пары")
    empty = wl.Watchlist(["FALLBACK_USDT"])
    wl.candidates = lambda: set()
    wl.portfolio = lambda: set()
    empty.refresh()
    check(empty.current == {"FALLBACK_USDT"}, "включается запасной список")

    print("\n9. Чтение настоящего файла состояния")
    tmp = Path(__file__).resolve().parent / "_watchtest_state.json"
    tmp.write_text(json.dumps({"positions": {"LONGXIA/USDT:USDT": {}, "HNT/USDT:USDT": {}}}))
    wl.BOT_STATE_PATH = tmp
    check(real_portfolio() == {"LONGXIA_USDT", "HNT_USDT"}, "позиции разобраны из файла")
    tmp.write_text("{}")
    check(real_portfolio() is None, "файл без positions читается как отказ, а не как пустота")
    tmp.write_text("не json")
    check(real_portfolio() is None, "битый файл читается как отказ")
    tmp.unlink()
    check(real_portfolio() is None, "отсутствующий файл читается как отказ")

    print("\nВСЁ ПРОШЛО")


if __name__ == "__main__":
    main()
