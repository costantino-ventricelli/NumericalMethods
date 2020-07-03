# -*- coding: utf-8 -*-
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
    __std_vandermonde_solution = np.array([1., 1., 1.])
    __std_hilbert_solution = np.array([1., 1., 1.])

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
        matrix = np.copy(coefficient_matrix)
        b_vector = np.copy(known_terms_vector)
        n, _ = np.shape(matrix)
        pivoting = PartialPivoting(matrix, b_vector)
        matrix, b_vector, _, _ = pivoting.get_results()
        determinant = np.linalg.det(matrix)
        if determinant != 0:
            for j in range(n - 1):
                for i in range(j + 1, n):
                    m = matrix[i, j] / matrix[j, j]
                    matrix[i, j] = 0
                    for k in range(j + 1, n):
                        matrix[i, k] = matrix[i, k] - (m * matrix[j, k])
                    b_vector[i] = b_vector[i] - (m * b_vector[j])
        return matrix, b_vector

    @staticmethod
    def __lu_factorization_method(coefficient_matrix, known_terms_vector):
        a_matrix = np.copy(coefficient_matrix)
        b_vector = np.copy(known_terms_vector)
        n, _ = np.shape(a_matrix)
        pivoting = PartialPivoting(a_matrix, b_vector)
        a_matrix, b_vector, _, exchange_array = pivoting.get_results()
        determinant = np.linalg.det(a_matrix)
        if determinant != 0:
            upper_matrix = np.copy(a_matrix)
            lower_matrix = np.zeros((n, n))
            for j in range(n - 1):
                lower_matrix[j, j] = 1
                for i in range(j + 1, n):
                    lower_matrix[i, j] = upper_matrix[i, j] / upper_matrix[j, j]
                    upper_matrix[i, j] = 0
                    for k in range(j + 1, n):
                        upper_matrix[k, i] = upper_matrix[k, i] - upper_matrix[j, k] * lower_matrix[i, j]
            lower_matrix[n - 1, n - 1] = 1
        else:
            upper_matrix = np.zeros((n, n))
            lower_matrix = np.zeros((n, n))
        return upper_matrix, lower_matrix, b_vector, exchange_array

    @staticmethod
    def std_vandermonde_matrix_gauss():
        coefficient_matrix, known_terms_vector = LinearTest.__gauss_elimination_method(
            LinearTest.__std_vandermonde_matrix,
            LinearTest.__vandermonde_known_value)
        substitution_algorithm = SubstitutionAlgorithm(coefficient_matrix, known_terms_vector)
        substitution_algorithm.backward_calculus()
        return substitution_algorithm.get_solution_vector(), \
               np.abs(np.linalg.norm(LinearTest.__std_vandermonde_solution -
                                     np.linalg.norm(substitution_algorithm.get_solution_vector()))), \
               np.abs(np.linalg.norm(LinearTest.__std_vandermonde_solution -
                                     np.linalg.norm(substitution_algorithm.get_solution_vector()))) / \
               np.linalg.norm(LinearTest.__std_vandermonde_solution)

    @staticmethod
    def std_hilbert_matrix_gauss():
        coefficient_matrix, known_terms_vector, = LinearTest.__gauss_elimination_method(
            LinearTest.__std_hilbert_matrix,
            LinearTest.__hilbert_known_value)
        substitution_algorithm = SubstitutionAlgorithm(coefficient_matrix, known_terms_vector)
        substitution_algorithm.backward_calculus()
        return substitution_algorithm.get_solution_vector(), \
               np.abs(np.linalg.norm(LinearTest.__std_hilbert_solution -
                                     np.linalg.norm(substitution_algorithm.get_solution_vector()))), \
               np.abs(np.linalg.norm(LinearTest.__std_hilbert_solution -
                                     np.linalg.norm(substitution_algorithm.get_solution_vector()))) / \
               np.linalg.norm(LinearTest.__std_hilbert_solution)

    @staticmethod
    def std_vandermonde_matrix_lu():
        upper_matrix, lower_matrix, known_terms_vector, exchange_array = LinearTest.__lu_factorization_method(
            LinearTest.__std_vandermonde_matrix, LinearTest.__vandermonde_known_value)
        print("UPPER MATRIX: \n", upper_matrix, "\nLOWER MATRIX: \n", lower_matrix)
        forward_substitution = SubstitutionAlgorithm(lower_matrix, known_terms_vector)
        forward_substitution.forward_calculus()
        backward_substitution = SubstitutionAlgorithm(upper_matrix, forward_substitution.get_solution_vector())
        backward_substitution.backward_calculus()
        solution_vector = [y for x, y in sorted(zip(exchange_array,
                                                    backward_substitution.get_solution_vector()))]
        return solution_vector, \
               np.abs(np.linalg.norm(LinearTest.__std_vandermonde_solution -
                                     np.linalg.norm(solution_vector))), \
               np.abs(np.linalg.norm(LinearTest.__std_vandermonde_solution -
                                     np.linalg.norm(solution_vector)) / \
                      np.linalg.norm(LinearTest.__std_vandermonde_solution))

    @staticmethod
    def std_hilbert_matrix_lu():
        upper_matrix, lower_matrix, known_terms_vector, exchange_array = LinearTest.__lu_factorization_method(
            LinearTest.__std_hilbert_matrix, LinearTest.__hilbert_known_value)
        print("UPPER MATRIX: \n", upper_matrix, "\nLOWER MATRIX: \n", lower_matrix)
        forward_substitution = SubstitutionAlgorithm(lower_matrix, known_terms_vector)
        forward_substitution.forward_calculus()
        backward_substitution = SubstitutionAlgorithm(upper_matrix, forward_substitution.get_solution_vector())
        backward_substitution.backward_calculus()
        solution_vector = [y for x, y in sorted(zip(exchange_array,
                                                    backward_substitution.get_solution_vector()))]
        return solution_vector, \
               np.abs(np.linalg.norm(LinearTest.__std_hilbert_solution -
                                     np.linalg.norm(solution_vector))), \
               np.abs(np.linalg.norm(LinearTest.__std_hilbert_solution -
                                     np.linalg.norm(solution_vector)) / \
                      np.linalg.norm(LinearTest.__std_vandermonde_solution))