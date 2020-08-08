# -*- coding: utf-8 -*-
"""
    Il primo metodo di interpolazione che andremo a studiare è il metodo polinomiale a coefficienti indeterminati, il
    quale genera una matrice Vandermonde per determinare il valore dei coefficienti i quali ci permettono poi di costruire
    il polinomio interpolante.
"""

import numpy as np
import matplotlib.pyplot as plot


class UnknownCoefficient:

    def __init__(self):
        # costruzione intervallo da 0 a 2pi
        self.__start_interval = 0
        self.__end_interval = 2 * np.pi
        # definizione grado polinomio interpolante
        self.__n = 15
        # generazione dei valori in x simulati
        self.__x_nodes = np.linspace(self.__start_interval, self.__end_interval, self.__n + 1)
        # calcolo dei valori di y, normali e con perturbazione
        self.__y_nodes = UnknownCoefficient.f(self.__x_nodes)
        self.__y_nodes_perturbed = UnknownCoefficient.f(self.__x_nodes) + ((np.random.rand(self.__n + 1)) * 1.0e-2)
        # generazione matrice di Vandermonde per il calocolo dei coefficienti
        vandermonde_matrix = [[self.__x_nodes[i] ** j for j in range(self.__n + 1)] for i in range(self.__n + 1)]
        # calcolo dei coefficienti con risoluzione sistema lienare
        coefficient = np.linalg.solve(vandermonde_matrix, self.__y_nodes)
        coefficient_perturbed = np.linalg.solve(vandermonde_matrix, self.__y_nodes_perturbed)
        # generazione dei valori necessari per effettuare il plot dei polinomi
        self.__x_val = np.linspace(self.__start_interval, self.__end_interval, 200)
        self.__polynomial_val = np.polyval(np.flipud(coefficient), self.__x_val)
        self.__polynomial_val_perturbed = np.polyval(np.flipud(coefficient_perturbed), self.__x_val)
        self.__fx = UnknownCoefficient.f(self.__x_val)

    @staticmethod
    def f(x):
        return np.cos(x) + np.sin(x)

    def plot_polynomial(self):
        plot.close('all')
        plot.figure(0)
        label_p = 'p_{%d}(x)' % self.__n
        label_f = "f(x)"
        plot.plot(self.__x_val, self.__polynomial_val, label=label_p)
        plot.plot(self.__x_val, self.__polynomial_val_perturbed, label="perturbed")
        plot.figure(1)
        plot.plot(self.__x_val, self.__fx, label=label_f)
        plot.show()
