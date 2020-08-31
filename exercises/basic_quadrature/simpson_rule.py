"""
    La regola di Simpson si basa sulla regola del trapezio con una precisione maggiore dato che utilizza tre punti nella
    funzione anzi che due.
"""

import numpy as np


def f(x):
    return 3 / 2 * np.sqrt(x) + 3 * x ** 2 + 1


def f_sec(x):
    return 6 - (3 / (8 * x ** (3 / 2)))


def F(x):
    return x ** (3 / 2) + x ** 3 + x + 2


class SimpsonRule:

    @staticmethod
    def simpson_classic(start, end):
        h = (end - start) / 2
        c = (start + end) / 2
        integral = F(end) - F(start)
        simpson = (h / 3) * (f(start) + 4 * f(c) + f(end))
        error = abs(integral - simpson)
        return integral, simpson, error

    @staticmethod
    def simpson_compose(start, end):
        m = np.amax(f_sec(np.linspace(start, end, 1000)))
        n = int(np.round(np.sqrt((((end - start) ** 5) * m) / (180 * 1.0e-5)), decimals=0))
        xi = np.linspace(start, end, n + 1)
        h = (end - start) / n
        integral = F(end) - F(start)
        first_sum = 0.00
        second_sum = 0.00
        stop = int(np.round((n / 2), decimals=0))
        for i in range(1, stop - 1):
            first_sum += f(xi[2 * i])
        for i in range(1, stop - 1):
            second_sum += f(xi[2 * i + 1])
        simpson = (h / 3) * ((f(xi[0]) - f(xi[n])) + (2 * first_sum) + (4 * second_sum))
        error = abs(integral - simpson)
        return integral, simpson, error, n
