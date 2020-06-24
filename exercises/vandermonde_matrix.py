"""
    La matrice di Vandermonde è una particolare matrice che contiene al suo interno una progressione geometrica di
    valri:
                alfa(1)^0, alfa(1)^1, ... , alfa(1)^n
                alfa(2)^0, alfa(2)^1, ... , alfa(2)^n
        A =     ...         ...       ...      ...
                alfa(i)^0, alfa(i)^1, ... , alfa(i)^n
                alfa(m)^0, alfa(m)^1, ... , alfa(m)^n

    In questo esperimento vogliamo verificare il numero di condizionamento K(A) della matrice di Vandermonde.
"""

import numpy as np
from random import random


class VandermondeMatrix:

    def __init__(self, dimension):
        self.__dimension = dimension
        self.__alfa = np.linspace(0, 1, dimension)
        self.__matrix = np.arange(float(dimension ** 2)).reshape(dimension, dimension)
        self.__condition_number = 0.00

    def calculate_condition_number(self):
        for i in range(self.__dimension):
            for j in range(self.__dimension):
                self.__matrix[i, j] = self.__alfa[i] ** j
        print(np.float128(np.linalg.det(self.__matrix)))
        self.__condition_number = np.linalg.cond(self.__matrix)

    def get_condition_number(self):
        return self.__condition_number

    def get_matrix(self):
        return self.__matrix

    def get_alfa(self):
        return self.__alfa
