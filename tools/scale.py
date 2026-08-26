"""Масштаб сигнала: скользящая оценка вместо замороженной.

Почему это не деталь. Порог «четыре сигмы» считался по первым 1000 узлам
сетки — то есть по 200 секундам записи — и дальше применялся на все 48 часов.
Сдвиг окна на несколько часов менял сигму целиком, а вместе с ней и выборку
сделок: у SOL их число падало с 863 до 138, у BTC с 564 до 165, PEPE
переворачивал знак. Два прогона с общими 87% данных давали разные ответы.
Такой тест меряет не рынок, а то, какая была волатильность в первые три
минуты окна.

Здесь масштаб продолжает обновляться на всём прогоне. Влияние начала окна
затухает с полупериодом в полчаса, а не остаётся навсегда.

Оценка идёт по СРЕДНЕМУ МОДУЛЮ, а не по среднему квадрату: у потока сделок
хвосты тяжёлые, и один выброс, возведённый в квадрат, задирает сигму так, что
следующие полчаса сигналов не будет вовсе. Для нормального распределения
sigma = sqrt(pi/2) * E|x| — этим коэффициентом и приводим.

Причинность соблюдена: оценка в момент t использует только прошлое, включая
сам t. Живой бот считал бы ровно так же.
"""

import numpy as np

SIGMA_FROM_ABS = float(np.sqrt(np.pi / 2))   # 1.2533
HALF_LIFE_S = 1800.0                          # полчаса
WARMUP_NODES = 300                            # минута при шаге 200 мс


class RollingSigma:
    """Потоковая оценка: узел за узлом, для бэктеста."""

    def __init__(self, step_ms, half_life_s=HALF_LIFE_S, warmup=WARMUP_NODES):
        step_s = step_ms / 1000.0
        self.alpha = 1.0 - 0.5 ** (step_s / half_life_s)
        self.warmup = warmup
        self.n = 0
        self.mean_abs = 0.0

    def update(self, x):
        """Добавить наблюдение и вернуть текущую сигму (None, пока разогрев)."""
        value = abs(float(x))
        if not np.isfinite(value):
            value = 0.0
        self.n += 1
        if self.n <= self.warmup:
            # на разогреве простое среднее: экспоненциальное на первых
            # наблюдениях почти не двигается и держит ноль слишком долго
            self.mean_abs += (value - self.mean_abs) / self.n
        else:
            self.mean_abs += self.alpha * (value - self.mean_abs)
        return self.value()

    def value(self):
        if self.n < self.warmup or self.mean_abs <= 0:
            return None
        return self.mean_abs * SIGMA_FROM_ABS


def rolling_sigma(x, step_ms, half_life_s=HALF_LIFE_S, warmup=WARMUP_NODES):
    """То же правило по готовому массиву — для инструментов, считающих сетку.

    Возвращает массив той же длины: в каждом узле оценка по данным до него
    включительно, NaN пока идёт разогрев. Одно правило на бэктест и на замер
    преимущества: иначе два инструмента считают разные «четыре сигмы».
    """
    out = np.full(len(x), np.nan)
    state = RollingSigma(step_ms, half_life_s, warmup)
    for i, value in enumerate(x):
        current = state.update(value)
        if current is not None:
            out[i] = current
    return out
