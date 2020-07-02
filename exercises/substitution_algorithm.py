"""
    L'algoritmo di sostituzione all'indietro permette di risolvere sistemi lineari che generano matrici triangolari
    superiori:
    a1,1x1 + a1,2x2 + a1,3x3 + a1,4x4 + ... + a1,nxn   = b1         a1,1    a1,2        a1,3        a1,4    ... a1,n
             a2,2x2 + a2,3x3 + a2,4x4 + ... + a2,nxn   = b2         0       a2,2        a2,3        a2,4    ... a2,n
                                        ...             ...    =    ...     ...         ...         ...     ...
                               an-1,n-1xn-1 + an-1,nxn = bn         0       0           0           an-1,an-1   an-1,n
                                              an,n     = bn         0       0           0           0           an,n
    L'algoritmo parte trovando per primo il temine isolato an,n successivamente "risale" le righe della matrice sostitu-
    endo alle x i valori trovati.
"""


import numpy as np


class SubstitutionAlgorithm:

    def __init__(self, matrix, known_terms):
        self.__matrix = matrix
        self.__known_terms = known_terms
        self.__n, _ = np.shape(self.__matrix)
        self.__x = np.zeros(self.__n)
        if np.shape(self.__matrix)[0] != np.shape(self.__matrix)[1]:
            raise NotSquareMatrix("La matrice inserita non è quadrata")
        if np.size(self.__known_terms) != self.__n:
            raise VectorMatrixNotMatch("La matrice dei coefficienti e il vettore dei termini noti non coincidono")

    """
        NOTA: I vari -1 nei for sono dovuti ad una forzatura, nel senso che per forzare python a considerare lo 0 ho
        dovurto forzarlo a -1
    """

    def backward_calculus(self):
        for i in range(self.__n):
            for j in range(i):
                if self.__matrix[i][j] != 0:
                    raise NonTriangularMatrix("La matrice inserita non è triangolare superiore")
        n = self.__n - 1
        self.__x[n] = self.__known_terms[n] / self.__matrix[n][n]
        for i in range(n - 1, -1, -1):
            sum = 0
            for j in range(n, i - 1, -1):
                sum = sum + self.__matrix[i][j] * self.__x[j]
            self.__x[i] = (self.__known_terms[i] - sum) / self.__matrix[i][i]

    def forward_calculus(self):
        for i in range(self.__n):
            for j in range(i + 1):
                if self.__matrix[i][j] != 0:
                    raise NonTriangularMatrix("La matrice inserita non è triangolare inferiore")
        self.__x[0] = self.__known_terms[0] / self.__matrix[0][0]
        for i in range(1, self.__n):
            sum = 0.0
            for j in range(i + 1):
                sum += self.__matrix[i][j] * self.__x[j]
            self.__x[i] = 1/self.__matrix[i][i] * (self.__known_terms - sum)

    def get_solution_vector(self):
        return self.__x


class NonTriangularMatrix(Exception):
    pass


class VectorMatrixNotMatch(Exception):
    pass


class NotSquareMatrix(Exception):
    pass
