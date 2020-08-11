# -*- coding: utf-8 -*-
"""
    L'interpolazione di Newton utilizza il metodo delle differenze divise.
    L'algoritmo ha un costo computazionale di n^2/2, l'algoritmo è adattivo, ovvero aggiungendo un data point al set
    non si ricalcola l'intero polinomio ma si aggiunge solo il punto al calcolo.
"""


import numpy as np
import matplotlib.pyplot as plot


class NewtonInterpolation:

    @staticmethod
    def f(x):
        return 1/(1 + 25*x**2)

    def __init__(self):
        start_interval = -5
        end_interval = 5
        self.__n = 15
        self.__x_nodes = np.linspace(start_interval, end_interval, self.__n + 1)
        self.__y_nodes = NewtonInterpolation.f(self.__x_nodes)
        self.__matrix = np.zeros((self.__n + 1, self.__n + 1))
        for i in range(self.__n + 1):
            for j in range(i + 1):

