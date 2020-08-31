"""
    La regola del trapezio permette di calcolare numericamente il valore di un integrale senza ricorrere a metodi
    analitici.
"""

import numpy as np


def f(x):
    return 3/2 * np.sqrt(x) + 3 * x ** 2 + 1


def f_sec(x):
    return 6 - (3/(8 * x ** (3 / 2)))


def F(x):
    return x ** (3/2) + x ** 3 + x + 2


class TrapezoidalRule:

    @staticmethod
    def trapezoidal_classic(start, end):
        h = (end - start)
        integral = F(end) - F(start)
        trapezoidal = h * ((f(start) + f(end)) / 2)
        error = abs(integral - trapezoidal)
        return integral, trapezoidal, error

    @staticmethod
    def trapezoidal_compose(start, end):
        m = np.amax(f_sec(np.linspace(start, end, 1000)))
        n = int(np.round(np.sqrt((((end - start) ** 3) * m) / (12 * 1.0e-5)), decimals=0))
        h = (end - start) / n
        integral = F(end) - F(start)
        xi = np.linspace(start, end, n + 1)
        sum = 0.00
        for i in range(1, n):
            sum += f(xi[i])
        trapezoidal = ((h / 2) * (f(xi[0]) + f(xi[n]))) + (h * sum)
        error = abs(integral - trapezoidal)
        return integral, trapezoidal, error, n
