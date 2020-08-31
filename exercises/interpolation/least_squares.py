"""
    L'approssimazione ai minimi quadrati viene usata per interpolare dati di partenza che potrebbero generare una
    funzione troppo complicata. Questo tipo di interpolazione genera una funzione che non passa per i punti, ma si
    avvicina.
"""

import numpy as np
import matplotlib.pyplot as plot


class LeastSquareApproximation:

    def __int__(self, n, start, end):
        self.__start = start
        self.__end = end
        self.__n = n
        self.__x = np.linspace(start, end, n)
        self.__y = 2 * self.__x + 1 + (np.random.rand(n + 1) + 0.5)
        matrix = np.zeros((2, 2))
        coefficient = np.zeros(2)
        matrix[0, 0] = np.sum(self.__x ** 2)
        matrix[0, 1] = np.sum(self.__x)
        matrix[1, 0] = matrix[0, 1]
        matrix[1, 1] = self.__n + 1
        coefficient[0] = np.sum(self.__x, self.__y)
        coefficient[1] = np.sum(self.__y)
        self.__matrix = matrix
        self.__coefficient = coefficient

    def linear_regression(self):
        solution = np.linalg.solve(self.__matrix, self.__coefficient)
        alpha = solution[0]
        beta = solution[1]
        x = np.linspace(self.__start, self.__end, 300)
        yx = alpha * x + beta
        plot.figure(1)
        plot.plot(x, yx, label='ax+b')
        plot.plot(x, self.__y, 'ro')
        plot.xlabel('x', fontsize=14)
        plot.ylabel('y', fontsize=14)
        plot.legend(loc='upper left')
        plot.show(block=False)
