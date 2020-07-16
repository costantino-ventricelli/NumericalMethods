# -*- coding: utf-8 -*-
"""
    Questo modulo contiene una classe che effettua il pivoting parziale di una matrice.
    La tecnica del pivoting permette di rendere più stabile l'esecuzione degli algoritmi di soluzione diretta dei sistemi
    lineari.
"""


import numpy as np


class PartialPivoting:

    def __init__(self, matrix, known_terms):
        self.__exchange = 0
        self.__matrix = matrix
        self.__known_terms = known_terms
        row, column = np.shape(self.__matrix)
        self.__exchange_array = np.arange(0, row, 1)
        self.__exchange = 0
        for i in range(row):
            max = self.__matrix[i, i]
            for j in range(i + 1, row):
                for k in range(i, i + 1):
                    if self.__matrix[j, k] > max:
                        self.__matrix[[i, j]] = self.__matrix[[j, i]]
                        self.__known_terms[i], self.__known_terms[j] = self.__known_terms[j], self.__known_terms[i]
                        self.__exchange_array[i], self.__exchange_array[j] = \
                            self.__exchange_array[j], self.__exchange_array[i]
                        self.__exchange += 1

    def get_results(self):
        return self.__matrix, self.__known_terms, self.__exchange, self.__exchange_array
