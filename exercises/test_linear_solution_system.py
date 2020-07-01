"""
    In questo modulo testeremo le differenze tra i vari metodi di risoluzione di:
    -   Eliminazione di Gauss;
    -   Fattorizzazione LU;
    -   Jacobi;
    -   Gauss-Seidel.
    Attraverso il calcolo di matrici mal condizionate come guella di Vandermonde e Hilbert.
"""


import numpy as np


class LinearTest:

    __n = 3
    __std_vandermonde_matrix = np.arange(float(__n ** 2)).reshape(__n, __n)
    __std_hilbert_matrix = np.arange(float(__n ** 2)).reshape(__n, __n)
    __std_alfa = np.linspace(0, 1, __n)
    __vandermonde_known_value = np.array([1.0, 1.75, 3.0])
    __hilbert_known_value = np.array([1.83333333, 1.08333333, 0.78333333])

    for i in range(__n):
        for j in range(__n):
            __std_vandermonde_matrix[i, j] = __std_alfa[i] ** j

    for i in range(__n):
        for j in range(__n):
            __std_hilbert_matrix[i, j] = 1 / i + (j + 1)

    def __init__(self, dimension):
        self.__dimension = dimension
        self.__alfa = np.linspace(0, 1, dimension)
        self.__matrix_vandermonde = np.arange(float(dimension ** 2)).reshape(dimension, dimension)
        self.__matrix_hilbert = np.arange(float(dimension ** 2)).reshape(dimension, dimension)
        for i in range(self.__dimension):
            for j in range(self.__dimension):
                self.__matrix_vandermonde[i, j] = self.__alfa[i] ** j
        for i in range(self.__dimension):
            for j in range(self.__dimension):
                self.__matrix_hilbert[i, j] = 1/i + (j + 1)

    def solve_standard_matrix(self):
        
