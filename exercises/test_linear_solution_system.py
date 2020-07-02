"""
    In questo modulo testeremo le differenze tra i vari metodi di risoluzione di:
    -   Eliminazione di Gauss;
    -   Fattorizzazione LU;
    -   Jacobi;
    -   Gauss-Seidel.
    Attraverso il calcolo di matrici mal condizionate come guella di Vandermonde e Hilbert.
"""

import numpy as np
from exercises.partial_pivoting import PartialPivoting
from exercises.substitution_algorithm import SubstitutionAlgorithm


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
            __std_hilbert_matrix[i, j] = 1 / (i + j + 1)

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
                self.__matrix_hilbert[i, j] = 1 / (i + j + 1)

    @staticmethod
    def __gauss_elimination_method(coefficient_matrix, known_terms_vector):
        print("Matrice prima dell'applicazione del pivoting\n", coefficient_matrix,
              "\nTermini noti prima del pivoting: ", known_terms_vector)
        n, _ = np.shape(coefficient_matrix)
        pivoting = PartialPivoting(coefficient_matrix, known_terms_vector)
        coefficient_matrix, known_terms_vector, exchange = pivoting.get_results()
        print("Matrice dopo l'applicazione del pivoting\n", coefficient_matrix,
              "\nTermini noti dopo del pivoting: ", known_terms_vector)
        determinant = np.linalg.det(coefficient_matrix)
        if (exchange % 2) != 0:
            determinant *= -1
        if determinant != 0:
            for j in range(n - 1):
                for i in range(j + 1, n):
                    m = coefficient_matrix[i, j] / coefficient_matrix[j, j]
                    coefficient_matrix[i, j] = 0
                    for k in range(j + 1, n):
                        coefficient_matrix[i, k] = coefficient_matrix[i, k] - (m * coefficient_matrix[j, k])
                    known_terms_vector[i] = known_terms_vector[i] - (m * known_terms_vector[j])
        return coefficient_matrix, known_terms_vector

    @staticmethod
    def std_vandermonde_matrix():
        coefficient_matrix, known_terms_vector = LinearTest.__gauss_elimination_method(
            LinearTest.__std_vandermonde_matrix,
            LinearTest.__vandermonde_known_value)
        substitution_algorithm = SubstitutionAlgorithm(coefficient_matrix, known_terms_vector)
        substitution_algorithm.backward_calculus()
        print("Gauss on matrix: \n", coefficient_matrix, "\nKnwon terms: ", known_terms_vector)
        print("Souluzione: ", np.array(substitution_algorithm.get_solution_vector()))

    @staticmethod
    def std_hilbert_matrix():
        coefficient_matrix, known_terms_vector, = LinearTest.__gauss_elimination_method(
            LinearTest.__std_hilbert_matrix,
            LinearTest.__hilbert_known_value)
        substitution_algorithm = SubstitutionAlgorithm(coefficient_matrix, known_terms_vector)
        substitution_algorithm.backward_calculus()
        print("Gauss on matrix: \n", coefficient_matrix, "\nKnwon terms: ", known_terms_vector)
        print("Souluzione: ", np.array(substitution_algorithm.get_solution_vector()))
