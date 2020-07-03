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
    __tolerance = 1.0e-10
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
    def __jacobi_iterative_method(coefficient_matrix, known_terms_vector):
        n, _ = np.shape(coefficient_matrix)
        current_x = np.zeros(n)
        epsilon = np.linalg.norm(known_terms_vector) * LinearTest.__tolerance
        max_iteration = n * 2
        stop = False
        index = 0
        while not stop and index < max_iteration:
            previous_x = np.copy(current_x)
            for i in range(n):
                sum = 0.0
                for j in range(0, n):
                    if j != i:
                        sum += coefficient_matrix[i][j] * previous_x[j]
                current_x[i] = (known_terms_vector[i] - sum) / coefficient_matrix[i][i]
                if np.abs(np.linalg.norm(previous_x) - np.linalg.norm(current_x)) < epsilon:
                    stop = True
            else:
                index += 1
        if not stop and (index + 1) >= max_iteration:
            return False, current_x
        else:
            return True, current_x

    @staticmethod
    def __gauss_seidel_iterative_method(coefficient_matrix, known_value_vector):
        n, _ = np.shape(coefficient_matrix)
        epsilon = np.linalg.norm(known_value_vector) * LinearTest.__tolerance
        max_iteration = n * 2
        current_x = np.zeros(n)
        next_x = np.zeros(n)
        stop = False
        index = 0
        while not stop and index < max_iteration:
            previous_x = np.copy(current_x)
            current_x = np.copy(next_x)
            for i in range(n):
                sum = 0.00
                for j in range(i - 1):
                    sum += coefficient_matrix[i][j]*current_x[j]
                for j in range(i + 1, n):
                    sum += coefficient_matrix[i][j]*previous_x[j]
                next_x[i] = (known_value_vector[i] - sum) / coefficient_matrix[i][i]
            if np.linalg.norm(known_value_vector - np.dot(coefficient_matrix, next_x)) < epsilon:
                stop = True
            else:
                index += 1
        if not stop and index > max_iteration:
            return False, next_x
        else:
            return True, next_x

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
               np.abs(np.linalg.norm(LinearTest.__std_vandermonde_solution - substitution_algorithm.get_solution_vector())) / \
               np.linalg.norm(LinearTest.__std_vandermonde_solution)

    @staticmethod
    def std_hilbert_matrix_gauss():
        coefficient_matrix, known_terms_vector, = LinearTest.__gauss_elimination_method(
            LinearTest.__std_hilbert_matrix,
            LinearTest.__hilbert_known_value)
        substitution_algorithm = SubstitutionAlgorithm(coefficient_matrix, known_terms_vector)
        substitution_algorithm.backward_calculus()
        return substitution_algorithm.get_solution_vector(), \
               np.abs(np.linalg.norm(LinearTest.__std_hilbert_solution - substitution_algorithm.get_solution_vector())), \
               np.abs(np.linalg.norm(LinearTest.__std_hilbert_solution -
                                     substitution_algorithm.get_solution_vector())) / \
               np.linalg.norm(LinearTest.__std_hilbert_solution)

    @staticmethod
    def std_vandermonde_matrix_lu():
        upper_matrix, lower_matrix, known_terms_vector, exchange_array = LinearTest.__lu_factorization_method(
            LinearTest.__std_vandermonde_matrix, LinearTest.__vandermonde_known_value)
        forward_substitution = SubstitutionAlgorithm(lower_matrix, known_terms_vector)
        forward_substitution.forward_calculus()
        backward_substitution = SubstitutionAlgorithm(upper_matrix, forward_substitution.get_solution_vector())
        backward_substitution.backward_calculus()
        solution_vector = [y for x, y in sorted(zip(exchange_array,
                                                    backward_substitution.get_solution_vector()))]
        return solution_vector, \
               np.abs(np.linalg.norm(LinearTest.__std_vandermonde_solution -
                                     np.linalg.norm(solution_vector))), \
               np.abs(np.linalg.norm(LinearTest.__std_vandermonde_solution - solution_vector) / \
                      np.linalg.norm(LinearTest.__std_vandermonde_solution))

    @staticmethod
    def std_hilbert_matrix_lu():
        upper_matrix, lower_matrix, known_terms_vector, exchange_array = LinearTest.__lu_factorization_method(
            LinearTest.__std_hilbert_matrix, LinearTest.__hilbert_known_value)
        forward_substitution = SubstitutionAlgorithm(lower_matrix, known_terms_vector)
        forward_substitution.forward_calculus()
        backward_substitution = SubstitutionAlgorithm(upper_matrix, forward_substitution.get_solution_vector())
        backward_substitution.backward_calculus()
        solution_vector = [y for x, y in sorted(zip(exchange_array,
                                                    backward_substitution.get_solution_vector()))]
        return solution_vector, \
               np.abs(np.linalg.norm(LinearTest.__std_hilbert_solution -
                                     np.linalg.norm(solution_vector))), \
               np.abs(np.linalg.norm(LinearTest.__std_hilbert_solution - solution_vector) /
                      np.linalg.norm(LinearTest.__std_vandermonde_solution))

    @staticmethod
    def std_jacobi():
        coefficient_matrix = -4 * np.diag(np.ones(10)) + np.diag(np.ones(9), 1) + np.diag(np.ones(9), -1)
        theoretical_solution = np.ones(10)
        known_value_vector = np.dot(coefficient_matrix, theoretical_solution)
        convergence, solution = LinearTest.__jacobi_iterative_method(coefficient_matrix, known_value_vector)
        return solution, np.abs(np.linalg.norm(solution) - np.linalg.norm(theoretical_solution)), \
               np.abs(np.linalg.norm(solution - theoretical_solution) / np.linalg.norm(solution))

    @staticmethod
    def std_gauss_seidel():
        coefficient_matrix = -4 * np.diag(np.ones(10)) + np.diag(np.ones(9), 1) + np.diag(np.ones(9), -1)
        theoretical_solution = np.ones(10)
        known_value_vector = np.dot(coefficient_matrix, theoretical_solution)
        convergence, solution = LinearTest.__gauss_seidel_iterative_method(coefficient_matrix, known_value_vector)
        return solution, np.abs(np.linalg.norm(solution) - np.linalg.norm(theoretical_solution)), \
               np.abs(np.linalg.norm(solution - theoretical_solution) / np.linalg.norm(solution))
