# -*- coding: utf-8 -*-
"""
    L'interpolazione di Newton utilizza il metodo delle differenze divise.
    L'algoritmo ha un costo computazionale di n^2/2, l'algoritmo è adattivo, ovvero aggiungendo un data point al set
    non si ricalcola l'intero polinomio ma si aggiunge solo il punto al calcolo.
"""


import numpy as np
import matplotlib.pyplot as plot
from exercises.interpolation.chebyshev_polynomial import ChebyshevNodes


class NewtonInterpolation:

    @staticmethod
    def f(x):
        return 1/(1 + 25*x**2)

    def __init__(self):
        start_interval = -5
        end_interval = 5
        self.__n = 15
        self.__x_nodes = np.linspace(start_interval, end_interval, 300)
        self.__y_nodes = NewtonInterpolation.f(self.__x_nodes)
        self.__matrix = np.zeros((self.__n, self.__n))
        for i in range(self.__n):
            self.__matrix[i, 0] = self.__y_nodes[i]
        for j in range(1, self.__n):
            for i in range(j, self.__n):
                if self.__x_nodes[j] - self.__x_nodes[i - 1] > 1.0e-14:
                    self.__matrix[i, j] = (self.__matrix[i, j - 1] - self.__matrix[i - 1, j - 1]) \
                                         / (self.__x_nodes[j] - self.__x_nodes[i - 1])
                else:
                    self.__matrix[i, j] = 0

    def __product(self, x):
        prod = 1.0
        for i in range(self.__n):
            prod *= (x - self.__x_nodes[i])
        return prod

    def approximate(self, x_value):
        polynomial = np.zeros(len(x_value))
        for i in range(len(x_value)):
            sum = 0.0
            prod = self.__product(x_value[i])
            for j in range(self.__n):
                sum += self.__matrix[j, j] * prod
            polynomial[i] = sum
        fx = NewtonInterpolation.f(x_value)
        plot.close('all')
        plot.figure(0)
        label_p = 'p_{%d}(x)' % self.__n
        label_f = "f(x)"
        plot.plot(x_value, polynomial, label=label_p)
        plot.plot(x_value, fx, label=label_f)
        plot.legend()
        plot.show()
