# -*- coding: utf-8 -*-
"""
    L'interpolazione di Newton utilizza il metodo delle differenze divise.
    L'algoritmo ha un costo computazionale di n^2/2, l'algoritmo è adattivo, ovvero aggiungendo un data point al set
    non si ricalcola l'intero polinomio ma si aggiunge solo il punto al calcolo.
"""


import numpy as np
import matplotlib.pyplot as plot


def f(x):
    return 1/(1 + 25*x**2)


class NewtonInterpolation:

    def __init__(self, x_nodes):
        self.__x_nodes = x_nodes
        self.__n = len(x_nodes)
        self.__y_nodes = f(x_nodes)
        self.__m = self.__calculate_matrix()

    def __calculate_matrix(self):
        m = np.zeros((self.__n, self.__n))
        x = self.__x_nodes
        for i in range(self.__n):
            m[i][0] = self.__y_nodes[i]
        for j in range(1, self.__n):
            for i in range(j, self.__n):
                if np.abs(x[j - 1] - x[i]) > 1.0e-14:
                    m[i][j] = (m[i][j - 1] - m[i - 1][j - 1]) / (x[j - 1] - x[i])
                else:
                    print("cancellazione")
                    m[i][j] = 0.00
        return m

    def approximate(self, x_aix):
        polynomial = np.zeros(300)
        x_aix = x_aix
        for i in range(len(polynomial)):
            for j in range(self.__n):
                polynomial[i] += self.__m[j][j] * self.__product(x_aix[i], j)
        self.plot_polynomial(polynomial, x_aix)

    def plot_polynomial(self, polynomial, x_aix):
        fx = f(x_aix)
        plot.close('all')
        plot.figure(0)
        label_p = 'p_{%d}(x)' % self.__n
        label_f = "f(x)"
        plot.plot(x_aix, polynomial, label=label_p)
        plot.plot(x_aix, fx, label=label_f)
        plot.legend()
        plot.show()

    def __product(self, value, stop):
        product = np.longdouble(1.0)
        for i in range(stop):
            product = product * (abs(value) - abs(self.__x_nodes[i]))
        return product
