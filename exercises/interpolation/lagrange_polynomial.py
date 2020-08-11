# -*- coding: utf-8 -*-
"""
    Il polinomio interpolante di Lagrange permette di legare direttamente i coefficienti ai valori di y(i), inoltre il
    costo computazionale di questo metodo è meno elevato a fronte di una maggiore instabilità numerica.
"""

import time

import matplotlib.pyplot as plot
import numpy as np


class LagrangePolynomial:
    first_machine_number = 1.0e-14

    @staticmethod
    def f(x):
        return np.sin(2 * x)

    def __init__(self):
        self.__start_interval = 0
        self.__end_interval = 2 * np.pi
        self.__n = 10
        self.__x_nodes = np.linspace(self.__start_interval, self.__end_interval, self.__n + 1, endpoint=True)
        self.__y_nodes = LagrangePolynomial.f(self.__x_nodes)
        self.__weight = np.zeros(self.__n + 1, dtype=float)
        matrix = np.zeros((self.__n + 1, self.__n + 1))
        for i in range(self.__n + 1):
            for j in range(self.__n + 1):
                if i == j:
                    matrix[i, j] = 1
                elif j > i:
                    matrix[i, j] = self.__x_nodes[i] - self.__x_nodes[j]
                else:
                    matrix[i, j] = -matrix[j, i]
        for i in range(self.__n + 1):
            self.__weight[i] = 1 / np.prod(matrix[i, :])

    def compute_first_barycentric_form(self):
        x_val = np.linspace(self.__start_interval, self.__end_interval, 300, endpoint=False)
        polynomial = np.zeros(300)
        start_time = time.time()
        for i in range(len(x_val)):
            check_nodes = abs(x_val[i] - self.__x_nodes) < 1.0e-14
            if True in check_nodes:
                indexes = np.where(check_nodes)
                index = indexes[0][0]
                polynomial[i] = self.__y_nodes[index]
            else:
                polynomial[i] = self.__compute_lagrange(x_val[i])
        end_time = time.time()
        print("Tempo per il calcolo del polinomio: ", end_time - start_time)
        fx = LagrangePolynomial.f(x_val)
        plot.close('all')
        plot.figure(0)
        label_p = 'p_{%d}(x)' % self.__n
        label_f = "f(x)"
        plot.plot(x_val, polynomial, label=label_p)
        plot.plot(x_val, fx, label=label_f)
        plot.legend()
        plot.show()
        LagrangePolynomial.__plot_rest(polynomial, fx, x_val)

    def __compute_lagrange(self, xi):
        psi = np.prod(xi - self.__x_nodes)
        sum = 0.0
        for i in range(self.__n + 1):
            sum += (self.__weight[i] * self.__y_nodes[i]) / (xi - self.__x_nodes[i])
        return psi * sum

    def compute_second_barycentric_form(self):
        x_val = np.linspace(self.__start_interval, self.__end_interval, 300, endpoint=False)
        polynomial = np.zeros(300)
        start_time = time.time()
        for i in range(len(x_val)):
            check_nodes = abs(x_val[i] - self.__x_nodes) < 1.0e-14
            if True in check_nodes:
                indexes = np.where(check_nodes)
                index = indexes[0][0]
                polynomial[i] = self.__y_nodes[index]
            else:
                numerator = 0.00
                denominator = 0.00
                for j in range(self.__n + 1):
                    numerator += (self.__weight[j] * self.__y_nodes[j]) / (x_val[i] - self.__x_nodes[j])
                for j in range(self.__n + 1):
                    denominator += self.__weight[j] / (x_val[i] - self.__x_nodes[j])
                polynomial[i] = numerator / denominator
        end_time = time.time()
        print("Tempo per il calcolo del polinomio: ", end_time - start_time)
        fx = LagrangePolynomial.f(x_val)
        plot.close('all')
        plot.figure(0)
        label_p = 'p_{%d}(x)' % self.__n
        label_f = "f(x)"
        plot.plot(x_val, polynomial, label=label_p)
        plot.plot(x_val, fx, label=label_f)
        plot.legend()
        plot.show()
        LagrangePolynomial.__plot_rest(polynomial, fx, x_val)

    @staticmethod
    def __plot_rest(polynomial, fx, x_val):
        r = np.abs(polynomial - fx)
        plot.figure(1)
        plot.semilogy(x_val, r, label="Resto")
        plot.legend()
        plot.show()
