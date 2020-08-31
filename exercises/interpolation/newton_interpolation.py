# -*- coding: utf-8 -*-
"""
    L'interpolazione di Newton utilizza il metodo delle differenze divise.
    L'algoritmo ha un costo computazionale di n^2/2, l'algoritmo è adattivo, ovvero aggiungendo un data point al set
    non si ricalcola l'intero polinomio ma si aggiunge solo il punto al calcolo.
"""

import numpy as np
import matplotlib.pyplot as plot


def f(x):
    return 1 / (1 + 25 * x ** 2)


class NewtonInterpolation:

    def __init__(self, x_nodes):
        self.__n = len(x_nodes)
        self.__x_nodes = x_nodes
        self.__y_nodes = f(self.__x_nodes)
        self.__matrix = np.zeros((self.__n, self.__n))
        for i in range(self.__n):
            self.__matrix[i][0] = self.__y_nodes[i]
        for i in range(1, self.__n):
            for j in range(self.__n - i):
                self.__matrix[j][i] = ((self.__matrix[j][i - 1] - self.__matrix[j + 1][i - 1]) /
                           (self.__x_nodes[j] - self.__x_nodes[i + j]))

    def plot_approximation(self):
        x_value = np.linspace(self.__x_nodes[0], self.__x_nodes[self.__n - 1], 300, endpoint=False)
        polynomial = np.zeros(len(x_value))
        for i in range(len(x_value)):
            for j in range(len(self.__x_nodes)):
                polynomial[i] += self.__matrix[0][j] * self.__product(j, x_value[i])
        plot.close('all')
        plot.figure(0)
        label_p = 'p_{%d}(x)' % self.__n
        label_f = "f(x)"
        plot.plot(x_value, polynomial, label=label_p)
        plot.plot(x_value, f(x_value), label=label_f)
        plot.legend()
        plot.show()

    def __product(self, i, value):
        pro = 1
        for j in range(i):
            pro = pro * (value - self.__x_nodes[j])
        return pro
