#!/usr/bin/env python3
"""Существует ли вообще прибыльная конструкция сделки — или их нет.

    python tools/sweep.py --symbol BTC_USDT --hours 24

Это НЕ подбор параметров. Подбор здесь бессмыслен и вреден: настроенный на
одном куске результат не переносится на следующий, это уже проверялось на
моментум-стратегиях. Здесь проверяется другое — структурный вопрос.

Арифметика такая: измеренное преимущество около 0.35 б.п. на сделку, а выход
по рынку стоит комиссию плюс переход через спред. Комиссия берётся с биржи
ПОИНСТРУМЕНТНО и у половины нашей выборки равна нулю (SOL, HYPE, XRP, PEPE,
ZEC), а у BTC доходит до 4 б.п. Плоская единица на всех, стоявшая здесь
раньше, была неверна в обе стороны сразу и на нулевых инструментах списывала
2 б.п. за круг из ничего.

Поэтому смысл столбца «доля лимитных выходов» разный: там, где комиссии нет,
рыночный выход стоит один спред, и пассивность перестаёт быть обязательной.

Поэтому в таблице главный столбец — не прибыль, а ДОЛЯ ЛИМИТНЫХ ВЫХОДОВ.
Прибыль без неё смысла не имеет.

Каждая конструкция считается на обеих половинах записи отдельно. Совпадение
знака на двух половинах — минимальное требование; несовпадение означает шум,
какой бы красивой ни была общая цифра.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.backtest import (replay_multi, replay_all, strategy,
                            reversal_strategy, delta_strategy,
                            random_control, taker_fee_bp,
                            taker_fee_spread)

# Конструкции выбраны из измеренной динамики, а не перебором:
# медиана движения за 60 с — 2.155 б.п., поэтому стоп в 2 б.п. стоит внутри
# шума; 999 означает «стопа нет», выход только по цели или по таймеру.
GRID = [
    #  имя                          стоп  цель  таймер  порог
    ("стоп 2 / цель 2 / 60с",         2,    2,     60,   0.4),
    ("стоп 6 / цель 3 / 60с",         6,    3,     60,   0.4),
    ("стоп 8 / цель 6 / 120с",        8,    6,    120,   0.4),
    ("без стопа / цель 2 / 30с",    999,    2,     30,   0.4),
    ("без стопа / цель 3 / 60с",    999,    3,     60,   0.4),
    ("без стопа / цель 5 / 300с",   999,    5,    300,   0.4),
    ("сильный сигнал, без стопа",   999,    3,     60,   0.7),
    ("сильный сигнал, стоп 6",        6,    3,     60,   0.7),
]


def split_stats(trades, cutoff_us):
    """Первая и вторая половина записи считаются раздельно."""
    out = []
    for chunk in ([t for t in trades if t.entry_us < cutoff_us],
                  [t for t in trades if t.entry_us >= cutoff_us]):
        if len(chunk) < 5:
            out.append(None)
            continue
        x = np.array([t.pnl_bp for t in chunk])
        t_stat = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if x.std() else 0.0
        out.append({"n": len(x), "mean": float(x.mean()), "t": float(t_stat)})
    return out


def anatomy(trades):
    """Откуда берётся результат: как выходили, сколько держали, как часто в плюс.

    Нужно затем, что средний результат в базисных пунктах ничего не говорит о
    механике. Если стратегия якобы забирает почти весь ход цены, ответ виден
    здесь: либо цель срабатывает подозрительно часто, либо сделки держатся
    секунды, либо выходов по стопу почти нет.
    """
    if not trades:
        return ""
    import numpy as np
    hold = np.array([(t.exit_us - t.entry_us) / 1e6 for t in trades])
    wins = sum(1 for t in trades if t.pnl_bp > 0) / len(trades) * 100
    by = {}
    for t in trades:
        by[t.exit_reason] = by.get(t.exit_reason, 0) + 1
    mix = " ".join(f"{k.split()[0]} {v*100//len(trades)}%" for k, v in
                   sorted(by.items(), key=lambda kv: -kv[1]))
    return (f"держали медиана {np.median(hold):.0f} с, в плюс {wins:.0f}%, "
            f"выходы: {mix}")


def maker_share(trades):
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.exit_reason == "цель лимиткой") / len(trades) * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTC_USDT",
                    help="инструмент или all — пройти по всем записываемым")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--lag-ms", type=int, default=200)
    ap.add_argument("--taker-bp", type=float, default=None,
                    help="комиссия тейкера; по умолчанию берётся с биржи "
                         "ПОИНСТРУМЕНТНО (половина нашей выборки — нулевая)")
    ap.add_argument("--notional", type=float, default=2500)
    ap.add_argument("--compare-order", action="store_true",
                    help="прогнать оба порядка времени и сравнить (одна команда)")
    ap.add_argument("--exch-time", action="store_true",
                    help="проигрывать в порядке биржевого времени, а не получения")
    ap.add_argument("--full", action="store_true",
                    help="гонять и заведомо мёртвые конструкции (медленнее втрое)")
    ap.add_argument("--profile", action="store_true",
                    help="показать, где уходит время (берите с --hours 2)")
    a = ap.parse_args()

    if getattr(a, "compare_order", False):
        compare_orders(a)
        return
    if a.profile:
        import cProfile, pstats, io as _io
        a.profile = False
        pr = cProfile.Profile(); pr.enable()
        main_body(a)
        pr.disable()
        st = pstats.Stats(pr); st.sort_stats("tottime")
        buf = _io.StringIO(); st.stream = buf; st.print_stats(14)
        print("\n  ГДЕ УХОДИТ ВРЕМЯ")
        print("\n".join(buf.getvalue().splitlines()[4:22]))
        return
    main_body(a)


def main_body(a):
    from config import SYMBOLS
    targets = SYMBOLS if a.symbol == "all" else [a.symbol]
    summary = []
    # Набор конструкций строится на КАЖДЫЙ инструмент отдельно: комиссия у них
    # разная, и общий набор молча списывал бы чужую ставку.
    runs_by = {t: build_runs(a, fee_for(a, t))[1] for t in targets}
    # base несёт только параметры чтения (часы, порядок, шаг сетки) — они у
    # всех инструментов общие.
    base = build_runs(a, fee_for(a, targets[0]))[0]
    if len(targets) > 1:
        # один проход по файлам на все инструменты: раньше каждый читал
        # запись заново, и перебор по восьми занимал полчаса
        results = replay_all(targets, base, runs_by)
        for target in targets:
            if target in results:
                report_one(target, a, results[target], runs_by[target], summary)
            else:
                print(f"\n  {target}: данных нет")
    else:
        for target in targets:
            report_one(target, a, replay_multi(target, base, runs_by[target]),
                       runs_by[target], summary)
    if len(targets) > 1:
        print("\n" + "=" * 96)
        print("  СВОДКА: лучшая конструкция по каждому инструменту")
        print("=" * 96)
        print(f"  {'инструмент':<14}{'спред б.п.':>11}{'конструкция':<30}"
              f"{'1-я пол.':>10}{'2-я пол.':>10}{'лимитных':>10}")
        for row in summary:
            print(f"  {row['symbol']:<14}{row['spread']:>11.3f}{row['name']:<30}"
                  f"{row['first']:>10.3f}{row['second']:>10.3f}{row['maker']:>9.0f}%")
        print("=" * 96)
        print("  Пассивная стратегия зарабатывает спред и платит за невыгодные")
        print("  исполнения. Чем шире спред в б.п., тем больше можно заработать —")
        print("  на BTC спред 0.013 б.п. и зарабатывать там нечего в принципе.")
        print("=" * 96)


def compare_orders(a):
    """Два прогона — по времени получения и по биржевому — и сравнение.

    Различаем ровно одно: врёт ли модель исполнения или книга с лентой
    разъехались во времени. Если доля невозможных исполнений при биржевом
    порядке падает — причина была в рассинхронизации, и результат в силе.
    Если не падает — наливаются заявки там, где рынка не было.
    """
    from config import SYMBOLS
    targets = SYMBOLS if a.symbol == "all" else [a.symbol]
    collected = {}
    for mode, label in (("local", "по получению"), ("exch", "по бирже")):
        a.exch_time = (mode == "exch")
        runs_by = {t: build_runs(a, fee_for(a, t))[1] for t in targets}
        base = build_runs(a, fee_for(a, targets[0]))[0]
        results = replay_all(targets, base, runs_by)
        rows = {}
        for sym in targets:
            res = results.get(sym)
            if not res:
                continue
            res.pop("__spread__", None)
            res.pop("__path__", None)
            bad = res.pop("__impossible__", {})
            res.pop("__samples__", None)
            best, best_name = None, None
            for name in res:
                if not name.startswith("поток"):
                    continue
                trades = res[name][0]
                if len(trades) < 20:
                    continue
                import numpy as np
                mean = float(np.mean([t.pnl_bp for t in trades]))
                if best is None or mean > best:
                    best, best_name = mean, name
            if best_name:
                trades = res[best_name][0]
                rows[sym] = {"n": len(trades), "mean": best,
                             "bad": bad.get(best_name, 0), "name": best_name}
        collected[mode] = rows
        print(f"  прогон {label}: готово", flush=True)

    print("\n" + "=" * 92)
    print("  СРАВНЕНИЕ ПОРЯДКА ВРЕМЕНИ")
    print("  Если доля невозможных исполнений падает при биржевом порядке —")
    print("  причина была в рассинхронизации ленты и стакана, а не в модели.")
    print("=" * 92)
    print(f"  {'инструмент':<13}{'сделок':>8}{'по получению':>26}{'по бирже':>26}")
    print(f"  {'':<13}{'':>8}{'б.п.':>12}{'невозм.':>14}{'б.п.':>12}{'невозм.':>14}")
    for sym in targets:
        L = collected.get("local", {}).get(sym)
        E = collected.get("exch", {}).get(sym)
        if not L and not E:
            continue
        def cell(x):
            if not x:
                return f"{'—':>12}{'—':>14}"
            share = x["bad"] / max(x["n"], 1) * 100
            return f"{x['mean']:>12.2f}{share:>13.0f}%"
        print(f"  {sym:<13}{(L or E)['n']:>8}{cell(L)}{cell(E)}")
    print("=" * 92)


def fee_for(a, symbol):
    """Комиссия тейкера для инструмента: с биржи, если её не задали руками.

    Плоская единица на все инструменты была неверна в обе стороны: BTC берёт
    2-4 б.п., а SOL, HYPE, XRP, PEPE и ZEC не берут ничего. На нулевых
    инструментах мы списывали 2 б.п. за круг, которых не существует.
    """
    return a.taker_bp if a.taker_bp is not None else taker_fee_bp(symbol)


def build_runs(a, taker_bp):
    base = {"hours": a.hours, "lag_ms": a.lag_ms, "step_ms": 200,
            "taker_bp": taker_bp, "size": 1.0, "order_life_s": 10,
            "stop_bp": 2, "target_bp": 2, "time_stop_s": 60,
            "imb_th": 0.4, "ofi_th": 0.0, "move_th": 3.0, "delta_th": 2.0,
            "order": "exch" if getattr(a, "exch_time", False) else "local"}

    runs = []
    # По умолчанию считаем только живое. Восемь пассивных конструкций и
    # четыре разворотных проверены и убыточны на всех инструментах и обеих
    # половинах — гонять их каждый раз значит втрое дольше ждать ради
    # подтверждения давно известного. Возвращаются флагом --full.
    grid = GRID if getattr(a, "full", False) else GRID[:1]
    for name, stop, target, timer, th in grid:
        params = dict(base, stop_bp=stop, target_bp=target,
                      time_stop_s=timer, imb_th=th)
        runs.append((name, params, strategy(params)))
    # конструкции на сигнале разворота: вход лимиткой против резкого движения
    for move_th, target, timer in ((3.0, 2, 60), (3.0, 3, 120),
                                   (5.0, 3, 120), (3.0, 5, 300)):
        params = dict(base, stop_bp=999, target_bp=target,
                      time_stop_s=timer, move_th=move_th)
        runs.append((f"разворот {move_th:g}сигм/цель {target}/{timer}с",
                     params, reversal_strategy(params)))
    # конструкции на потоке сделок: горизонт минут, цели в разы больше.
    # Раньше все цели были 2-6 б.п. при удержании минуту — там издержки
    # съедали всё независимо от сигнала.
    if getattr(a, "full", False):
        for delta_th, target, stop, timer in ((2.0, 8, 16, 300), (3.0, 8, 16, 300),
                                              (3.0, 15, 25, 900), (2.0, 15, 25, 900)):
            params = dict(base, stop_bp=stop, target_bp=target,
                          time_stop_s=timer, delta_th=delta_th)
            runs.append((f"поток {delta_th:g}сигм/цель {target}/{timer//60}мин",
                         params, delta_strategy(params)))
    # То же по потоку, но вход ПО РЫНКУ. Пассивно войти в моментум нельзя:
    # если покупатели забирают по аску, лимитку на биде исполнят только когда
    # цена вернётся, то есть исключительно на неудачных сигналах.
    for delta_th, target, stop, timer in ((2.0, 8, 16, 300), (3.0, 8, 16, 300),
                                          (3.0, 15, 25, 900), (4.0, 20, 30, 900)):
        params = dict(base, stop_bp=stop, target_bp=target, time_stop_s=timer,
                      delta_th=delta_th, taker_entry=True)
        runs.append((f"поток ПО РЫНКУ {delta_th:g}с/цель {target}/{timer//60}м",
                     params, delta_strategy(params)))
    # ЧИСТО РЫНОЧНЫЕ КОНСТРУКЦИИ: и вход, и выход по рынку.
    # Модель исполнения лимиток на этих данных недостоверна — биржевые метки
    # идут в миллисекундах, стакан приходит пачками по 200 мс, и внутри пачки
    # порядок событий восстановить нечем. Треть исполнений оказывается по
    # недостижимой цене даже при правильном порядке времени. Рыночные заявки
    # этого не касаются: они исполняются всегда и по известной цене, поэтому
    # результат такой конструкции ничем не завышен.
    for delta_th, target, stop, timer in ((3.0, 8, 16, 300), (4.0, 12, 20, 600),
                                          (4.0, 20, 30, 900), (5.0, 25, 35, 900)):
        params = dict(base, stop_bp=stop, target_bp=target, time_stop_s=timer,
                      delta_th=delta_th, taker_entry=True, taker_exit=True)
        runs.append((f"ТОЛЬКО РЫНОК {delta_th:g}с/цель {target}/{timer//60}м",
                     params, delta_strategy(params)))
    # БЕЗ БАРЬЕРОВ: вход по рынку, выход по рынку строго по таймеру. Ни стопа,
    # ни цели.
    #
    # Это перекрёстная проверка к tools/edge.py, который меряет ровно такую
    # сделку, но другим способом — по всем узлам сетки сразу, а не по сделкам
    # свободного движка. Там на HYPE вышло +2.68 б.п. на горизонте 5 минут при
    # пороге 3 сигмы. Здесь то же самое считает второй, независимо написанный
    # движок: с очередью, задержкой в 200 мс и реальными ценами касания вместо
    # середины. Сойдутся — находке можно верить. Разойдутся — один из двух
    # инструментов врёт, и надо искать который.
    #
    # Барьеры убраны намеренно: стоп 16 и цель 8 срезают распределение и
    # меняют измеряемую величину. Прежние прогоны sweep поэтому и не
    # сопоставимы с edge — там ещё и сигма была замороженной.
    for delta_th, timer in ((2.0, 60), (3.0, 60), (2.0, 300), (3.0, 300)):
        params = dict(base, stop_bp=999, target_bp=999, time_stop_s=timer,
                      delta_th=delta_th, taker_entry=True, taker_exit=True)
        runs.append((f"БЕЗ БАРЬЕРОВ {delta_th:g}σ/{timer//60} мин",
                     params, delta_strategy(params)))
    for timer in (60, 300):
        params = dict(base, stop_bp=999, target_bp=999, time_stop_s=timer,
                      taker_entry=True, taker_exit=True)
        runs.append((f"   контроль без барьеров {timer//60} мин",
                     params, random_control(0.004)))
    for delta_th, target, stop, timer in ((3.0, 8, 16, 300), (4.0, 20, 30, 900)):
        params = dict(base, stop_bp=stop, target_bp=target, time_stop_s=timer,
                      taker_entry=True, taker_exit=True)
        runs.append((f"   контроль рынок цель {target}/{timer//60}м",
                     params, random_control(0.004)))

    # СОГЛАСОВАННЫЙ КОНТРОЛЬ. Обычного случайного входа мало: цель 8 при стопе
    # 16 — асимметричный барьер, и он даёт высокий винрейт сам по себе, без
    # всякого сигнала. Поэтому к каждой конструкции по потоку идёт двойник с
    # теми же целью, стопом, таймером и входом по рынку, но со случайным
    # направлением. Разница между ними и есть вклад сигнала.
    for delta_th, target, stop, timer in ((2.0, 8, 16, 300), (3.0, 8, 16, 300),
                                          (3.0, 15, 25, 900), (4.0, 20, 30, 900)):
        params = dict(base, stop_bp=stop, target_bp=target, time_stop_s=timer,
                      taker_entry=True)
        runs.append((f"   контроль цель {target}/{timer//60}м",
                     params, random_control(0.004)))
    control = dict(base, stop_bp=999, target_bp=3, time_stop_s=60)
    runs.append(("СЛУЧАЙНЫЙ ВХОД", control, random_control(0.01)))

    return base, runs


def _fee_note(symbol, fee):
    if fee == 0:
        return " (БЕЗ КОМИССИИ)"
    lo, hi = taker_fee_spread(symbol)
    if lo is not None and hi != lo:
        return f" (биржа отвечала от {lo:g} до {hi:g} — взята худшая)"
    return ""


def report_one(symbol, a, result, runs, summary):
    # Порядок времени печатается наравне с остальными условиями: именно он
    # превращал +9.6 б.п. в -0.15, и без пометки сохранённый отчёт нельзя
    # прочитать позже — приходится опознавать прогон по доле невозможных
    # исполнений.
    order = "БИРЖЕВОЙ" if getattr(a, "exch_time", False) else "по получению"
    print("\n" + "=" * 96)
    fee = fee_for(a, symbol)
    print(f"  {symbol}   последние {a.hours:g} ч   порядок {order}   "
          f"задержка {a.lag_ms} мс   "
          f"тейкер {fee:g} б.п.{_fee_note(symbol, fee)}   "
          f"номинал ${a.notional:,.0f}")
    print("=" * 96)

    spread_med, _ = result.pop("__spread__")
    path_bp = result.pop("__path__", 0.0)
    impossible = result.pop("__impossible__", {})
    samples = result.pop("__samples__", {})
    print(f"  медианный спред {spread_med:.3f} б.п. — столько зарабатывает "
          f"круг «купил по биду, продал по аску»")

    all_trades = [t for trades, _ in result.values() for t in trades]
    if not all_trades:
        print("  ни одна конструкция не совершила сделок")
        return
    lo = min(t.entry_us for t in all_trades)
    hi = max(t.entry_us for t in all_trades)
    cutoff = lo + (hi - lo) / 2

    print(f"\n  {'конструкция':<28}{'сделок':>7}{'лимитных':>10}"
          f"{'1-я половина':>22}{'2-я половина':>22}")
    print(f"  {'':<28}{'':>7}{'выходов':>10}"
          f"{'ср.б.п.':>11}{'t':>11}{'ср.б.п.':>11}{'t':>11}")
    print("  " + "-" * 92)
    best = None
    for name, _, _ in runs:
        trades, _ = result[name]
        first, second = split_stats(trades, cutoff)
        row = f"  {name:<28}{len(trades):>7}{maker_share(trades):>9.0f}%"
        for half in (first, second):
            row += (f"{half['mean']:>11.3f}{half['t']:>11.2f}"
                    if half else f"{'—':>11}{'—':>11}")
        print(row)
        if first and second and name != "СЛУЧАЙНЫЙ ВХОД":
            worst = min(first["mean"], second["mean"])
            if best is None or worst > best[0]:
                best = (worst, {"symbol": symbol, "spread": spread_med,
                                "name": name, "first": first["mean"],
                                "second": second["mean"],
                                "maker": maker_share(trades)})
    if best:
        summary.append(best[1])
        name = best[1]["name"]
        print(f"\n    разбор лучшей ({name}): {anatomy(result[name][0])}")
        trades = result[name][0]
        taken = sum(abs(tr.pnl_bp) for tr in trades)
        share = taken / path_bp * 100 if path_bp else 0
        flag = "  <-- НЕВОЗМОЖНО, ищи ошибку" if share > 30 else ""
        print(f"    забрано {taken:,.0f} б.п. при полном ходе цены "
              f"{path_bp:,.0f} — это {share:.1f}%{flag}")
        bad = impossible.get(name, 0)
        if bad:
            print(f"    ИСПОЛНЕНИЙ ПО НЕДОСТИЖИМОЙ ЦЕНЕ: {bad} из {len(trades)} "
                  f"({bad/max(len(trades),1)*100:.0f}%)")
            for ex in samples.get(name, [])[:2]:
                print("      " + ", ".join(f"{k} {v}" for k, v in ex.items()))

    print("=" * 96)
    print("  Читать так: конструкция чего-то стоит, только если ср.б.п. положительно")
    print("  на ОБЕИХ половинах и обе t выше 3. Плюс на одной половине и минус на")
    print("  другой — это шум. Доля лимитных выходов ниже 70% означает, что комиссия")
    print("  съест преимущество независимо от остальных цифр.")
    print("=" * 96)
    return


if __name__ == "__main__":
    main()
