# -*- coding: utf-8 -*-
import time

import numpy as np
from tabulate import tabulate


from exercises.linear_system.linear_system_solver import LinearTest
from exercises.linear_system.rotation import Rotation
from exercises.linear_system.substitution_algorithm import SubstitutionAlgorithm
from exercises.linear_system.vandermonde_matrix import VandermondeMatrix
from exercises.machine_number.epsilon_machine import EpsilonMachine
from exercises.machine_number.function_conditioning import FunctionConditioning
from exercises.machine_number.product_conditioning import ProductConditioning
from exercises.machine_number.product_sum_test import ProductSumTest
from exercises.machine_number.real_min_max import RealMinMax
from exercises.machine_number.sum_conditioning import SumConditioning
from exercises.unlinear_equation.zero_methods import ZeroMethods
from exercises.interpolation.lagrange_polynomial import LagrangePolynomial
from exercises.interpolation.unknown_coefficient import UnknownCoefficient
from exercises.interpolation.newton_interpolation import NewtonInterpolation
from exercises.interpolation.chebyshev_polynomial import ChebyshevNodes
from exercises.interpolation.least_squares import LeastSquaresApproximation
from exercises.basic_quadrature.trapezoidal_rule import TrapezoidalRule
from exercises.basic_quadrature.simpson_rule import SimpsonRule


def epsilon_machine_calculus():
    epsilon_machine = EpsilonMachine()
    epsilon_machine.find_epsilon()
    epsilon_machine.find_precision()
    epsilon_machine.find_significant_digits()
    print("Epsilon machine: ", epsilon_machine.get_epsilon())
    print("Machine precision: ", epsilon_machine.get_precision())
    print("Significant digits: ", epsilon_machine.get_significant_digits())


def real_min_max_calculus():
    array_of_bit = [64, 32, 16, 0]
    for item in array_of_bit:
        double_precision = RealMinMax(item)
        double_precision.calculate_max_min_epsilon()
        print("Real max: ", double_precision.get_real_max())
        print("Real min: ", double_precision.get_real_min())
        print("Epsilon: ", double_precision.get_real_epsilon(), "\n")


def sum_conditioning():
    print("Inserisci il primo valore: ")
    first_number = float(input())
    print("Inserisci il secondo valore: ")
    second_number = float(input())
    print("Inserisci la perturbazione: ")
    perturbation = float(input())
    problem = SumConditioning(first_number, second_number, perturbation)
    print("Errore assoluto prodotto: %15.12f" % problem.get_absolute_error_sum())
    print("Errore relativo prodotto: %15.12f" % problem.get_relative_error_sum())
    print("Errore assoluto primo numero: %15.12f" % problem.get_first_number_absolute_error())
    print("Errore relativo primo numero: %15.12f" % problem.get_second_number_relative_error())
    print("Errore assoluto secondo numero: %15.12f" % problem.get_second_number_absolute_error())
    print("Errore relativo secondo numero: %15.12f" % problem.get_second_number_relative_error())


def product_conditioning():
    print("Inserisci il primo valore: ")
    first_number = float(input())
    print("Inserisci il secondo valore: ")
    second_number = float(input())
    print("Inserisci la perturbazione: ")
    perturbation = float(input())
    problem = ProductConditioning(first_number, second_number, perturbation)
    print("Errore assoluto somma: %15.12f" % problem.get_absolute_error_product())
    print("Errore relativo somma: %15.12f" % problem.get_relative_error_product())
    print("Errore assoluto primo numero: %15.12f" % problem.get_first_number_absolute_error())
    print("Errore relativo primo numero: %15.12f" % problem.get_second_number_relative_error())
    print("Errore assoluto secondo numero: %15.12f" % problem.get_second_number_absolute_error())
    print("Errore relativo secondo numero: %15.12f" % problem.get_second_number_relative_error())


def function_conditioning():
    print("Inserisci il valore di x0 per il quale vuoi calcolare le due funzioni\n"
          "- radice\n"
          "- tangente\n")
    number = input("Inserisci: ")
    perturbation = input("Inserisci il valore della perturbazione per x0: ")
    problem = FunctionConditioning(float(number), float(perturbation))
    print("Errore assoluto della funzione radice: %15.12f" % problem.get_absolute_error_square())
    print("Errore relativo della funzione radice: %15.12f" % problem.get_relative_error_square())
    print("Errore assoluto della funzione tangente: %15.12f" % problem.get_absolute_error_tan())
    print("Errore relativo della funzione tangente: %15.12f" % problem.get_relative_error_tan())


def sum_product_test():
    product_sum_test = ProductSumTest(input("Inserisci il numero di volte per cui calcolare i test: "))
    product_sum_test.calculate_machine_products_sum()
    product_sum_test.calculate_real_products_sum()
    print("Il valore delle somme in doppia precisione per il numero macchina è: ",
          product_sum_test.get_machine_double_sum(), " mentre il prodotto è: ",
          product_sum_test.get_machine_double_product())
    print("Il valore delle somme in doppia precisione per il numero reale è: ",
          product_sum_test.get_real_double_sum(), " mentre il prodotto è: ", product_sum_test.get_real_double_product(),
          "\n")
    print("Il valore delle somme in singola precisione per il numero macchina è: ",
          product_sum_test.get_machine_single_sum(), " mentre il prodotto è: ",
          product_sum_test.get_machine_single_product())
    print("Il valore delle somme in singola precisione per il numero reale è: ",
          product_sum_test.get_real_single_sum(), " mentre il prodotto è: ", product_sum_test.get_real_single_product(),
          "\n")
    print("Il valore delle somme in mezza precisione per il numero macchina è: ",
          product_sum_test.get_machine_half_sum(), " mentre il prodotto è: ",
          product_sum_test.get_machine_half_product())
    print("Il valore delle somme in mezza precisione per il numero reale è: ",
          product_sum_test.get_real_half_sum(), "mentre il prodotto è: ", product_sum_test.get_real_half_product(),
          "\n")


def rotate_image():
    rotation = Rotation(int(input("Inserisci l'angolo di rotazione dell'immagine: ")))
    rotation.rotate()


def vandermonde_condition():
    vandermonde = VandermondeMatrix(int(input("Inserisci la dimensione della mantrice: ")))
    vandermonde.calculate_condition_number()
    print("Il numero di condizione K(A) per la matrice: \n"
          , vandermonde.get_matrix(), "\nè: ",
          vandermonde.get_condition_number(), "\nL'alfa della mantrice è: ", vandermonde.get_alfa())


def forward_substitution_algorithm():
    dimension = int(input("Inserisci la dimensione della matrice: "))
    coefficients_matrix = np.zeros(dimension ** 2).reshape(dimension, dimension)
    for i in range(dimension):
        coefficients_matrix[i] = list(map(float, input("Inserisci la " + str(i + 1) + " riga: ").split()))
    known_terms = list(map(float, input("Inserisci il vettore dei coefficienti: ").split()))
    try:
        forward_substitution = SubstitutionAlgorithm(coefficients_matrix, known_terms)
        forward_substitution.backward_calculus()
        print("Il risultato del calcolo è: ", forward_substitution.get_solution_vector())
    except Exception as ex:
        print(ex)


def linear_system():
    linear_test = LinearTest(int(input("Inserisci la dimensione della matrice: ")))
    print("VANDERMONDE MATRIX CON METODO DI GAUSS")
    start_time = time.time()
    solution, absolute, relative = linear_test.std_vandermonde_matrix_gauss()
    total_time = (time.time() - start_time)
    print("Tempo di esecuzione: ", total_time)
    print("Soluzione: ", solution, "\nErrore assoluto: ", absolute, "\nRelative: ", relative, "\n")
    print("HILBERT MATRIX CON METODO DI GAUSS")
    start_time = time.time()
    solution, absolute, relative = linear_test.std_hilbert_matrix_gauss()
    total_time = (time.time() - start_time)
    print("Tempo di esecuzione: ", total_time)
    print("Soluzione: ", solution, "\nErrore assoluto: ", absolute, "\nRelative: ", relative, "\n")
    print("VANDERMONDE MATRIX CON FATTORIZZAZIONE LU")
    start_time = time.time()
    solution, absolute, relative = linear_test.std_vandermonde_matrix_lu()
    total_time = (time.time() - start_time)
    print("Tempo di esecuzione: ", total_time)
    print("Soluzione: ", solution, "\nErrore assoluto: ", absolute, "\nRelative: ", relative, "\n")
    print("HILBERT MATRIX CON FATTORIZZAZIONE LU")
    start_time = time.time()
    solution, absolute, relative = linear_test.std_hilbert_matrix_lu()
    total_time = (time.time() - start_time)
    print("Tempo di esecuzione: ", total_time)
    print("Soluzione: ", solution, "\nErrore assoluto: ", absolute, "\nRelative: ", relative, "\n")
    print("METODO ITERATIVO DI JACOBI")
    start_time = time.time()
    solution, absolute, relative = linear_test.std_jacobi()
    total_time = time.time() - start_time
    print("Tempo di esecuzione: ", total_time)
    print("Soluzione: ", solution, "\nErrore assoluto: ", absolute, "\nRelative: ", relative, "\n")
    print("METODO ITERATIVO DI GAUSS-SEIDEL")
    start_time = time.time()
    solution, absolute, relative = linear_test.std_gauss_seidel()
    total_time = time.time() - start_time
    print("Tempo di esecuzione: ", total_time)
    print("Soluzione: ", solution, "\nErrore assoluto: ", absolute, "\nRelative: ", relative, "\n")


def equation_solver():
    first_equation = np.array([2, 2, 0, 3], dtype=float)
    interval = np.array([-2, -1], dtype=int)
    first_bisection_time = time.time()
    first_bisection = ZeroMethods.bisection_method(first_equation, interval, 0)
    first_bisection_time = time.time() - first_bisection_time
    first_newton_time = time.time()
    first_newton = ZeroMethods.newton_method(first_equation, interval)
    first_newton_time = time.time() - first_newton_time
    first_secant_time = time.time()
    first_secant = ZeroMethods.secant_method(first_equation, interval)
    first_secant_time = time.time() - first_secant_time
    second_equation = np.array([3, 9, 5, -4, 6, -3, 2, -7, 8, 1, -3], dtype=float)
    interval = np.array([-1, 0], dtype=int)
    second_bisection_time = time.time()
    second_bisection = ZeroMethods.bisection_method(second_equation, interval, 0)
    second_bisection_time = time.time() - second_bisection_time
    second_newton_time = time.time()
    second_newton = ZeroMethods.newton_method(second_equation, interval)
    second_newton_time = time.time() - second_newton_time
    second_secant_time = time.time()
    second_secant = ZeroMethods.secant_method(second_equation, interval)
    second_secant_time = time.time() - second_secant_time
    print("\nEquazione: 2x^3+2x^2+3")
    table = [["Bisezione", first_bisection_time, first_bisection, np.abs(-1.59190088744889310 - first_bisection),
             np.abs(-1.59190088744889310 - first_bisection) / np.abs(first_bisection)],
             ["Newton", first_newton_time, first_newton, np.abs(-1.59190088744889310 - first_newton),
             np.abs(-1.59190088744889310 - first_newton) / np.abs(first_newton)],
             ["Secanti", first_secant_time, first_secant, np.abs(-1.59190088744889310 - first_secant),
             np.abs(-1.59190088744889310 - first_secant) / np.abs(first_secant)]]
    print(tabulate(table, headers=["Metodo", "Tempo di esecuzione", "Soluzione", "Errore assoluto", "Errore relativo"]))

    print("\nEquazione: 3x^10+9x^9+5x^8-4x^7+6x^6-3x^5+2x^4-7x^3+8x^2+x-3")
    table = [["Bisezione", second_bisection_time, second_bisection, np.abs(-0.5168670824290528 - second_bisection),
              np.abs(-0.5168670824290528 - second_bisection) / np.abs(second_bisection)],
             ["Newton", second_newton_time, second_newton, np.abs(-0.5168670824290528 - second_newton),
              np.abs(-0.5168670824290528 - second_newton) / np.abs(second_newton)],
             ["Secanti", second_secant_time, second_secant, np.abs(-0.5168670824290528 - second_secant),
              np.abs(-0.5168670824290528 - second_secant) / np.abs(second_secant)]]
    print(tabulate(table, headers=["Metodo", "Tempo di esecuzione", "Soluzione", "Errore assoluto", "Errore relativo"]))


def unknown_coefficient_interpolation():
    start_time = time.time()
    unknown_coefficient = UnknownCoefficient()
    end_time = time.time()
    print("Interpolazione calcolata in: ", (end_time - start_time))
    unknown_coefficient.plot_polynomial()


def lagrange_interpolation():
    choose = int(input("1) Prima forma di bisezione\n"
                   "2) Seconda forma di bisezione\n"
                   "Inserisci la scelta: "))
    lagrange = LagrangePolynomial()
    if choose == 1:
        lagrange.compute_first_barycentric_form()
    elif choose == 2:
        lagrange.compute_second_barycentric_form()
    else:
        print("Selezione non valida")


def newton_interpolation():
    newton = NewtonInterpolation(np.linspace(-1, 1, 5))


def newton_interpolation_chebyshev():
    nodes = ChebyshevNodes.get_chebyshev_nodes(-1, 1, 20)
    newton = NewtonInterpolation(nodes)
    newton.plot_approximation()


def least_squares():
    least_squares_approximation = LeastSquaresApproximation(200, -5, 2)
    least_squares_approximation.linear_regression()


def trapezoidal_rule():
    integral, trapezoidal, error = TrapezoidalRule.trapezoidal_classic(1, 4)
    print("Valore dell'integrale analitico: ", integral)
    print("Valore dell'integrale con regola classica: ", trapezoidal)
    print("Errore: ", error)
    integral, trapezoidal, error, n = TrapezoidalRule.trapezoidal_compose(1, 4)
    print("Valore dell'integrale analitico: ", integral)
    print("Valore dell'integrale con regola composta: ", trapezoidal)
    print("Errore: ", error)
    print("Numero cicli per ottenere il risultato: ", n)


def simpson_rule():
    integral, simpson, error = SimpsonRule.simpson_classic(1, 4)
    print("Valore dell'integrale analitico: ", integral)
    print("Valore dell'integrale con regola classica: ", simpson)
    print("Errore: ", error)
    integral, simpson, error, n = SimpsonRule.simpson_compose(1, 4)
    print("Valore dell'integrale analitico: ", integral)
    print("Valore dell'integrale con regola composta: ", simpson)
    print("Errore: ", error)
    print("Numero cicli per ottenere il risultato: ", n)


def main():
    print("Inserisci il numero del problema da richiamare:\n"
          "1)  Condizionamento somma;\n"
          "2)  Condizionamento prodotto;\n"
          "3)  Condizionamento calcolo funzioni;\n"
          "4)  Epsilon machine;\n"
          "5)  Minimo e massimo macchina;\n"
          "6)  Somma prodotto differenza;\n"
          "7)  Ruota immagine;\n"
          "8)  Matrice di Vandermonde;\n"
          "9)  Algoritmo si sostituzione all'indietro\n"
          "10) Sistemi lineari\n"
          "11) Equazioni non lineari\n"
          "12) Interpolazione a coefficienti ignoti\n"
          "13) Interpolazione di lagrange\n"
          "14) Interpolazione di Newton\n"
          "15) Interpolazione di Newton con Chebyshev\n"
          "16) Regressione lineare\n"
          "17) Regola del trapezio\n"
          "18) Regola di Simpson")
    print("Inserisci la tua scelta: ")
    switch(int(input()))


def switch(select):
    switcher = {
        1: sum_conditioning,
        2: product_conditioning,
        3: function_conditioning,
        4: epsilon_machine_calculus,
        5: real_min_max_calculus,
        6: sum_product_test,
        7: rotate_image,
        8: vandermonde_condition,
        9: forward_substitution_algorithm,
        10: linear_system,
        11: equation_solver,
        12: unknown_coefficient_interpolation,
        13: lagrange_interpolation,
        14: newton_interpolation,
        15: newton_interpolation_chebyshev,
        16: least_squares,
        17: trapezoidal_rule,
        18: simpson_rule
    }
    function = switcher.get(select, lambda: "Selezione non valida")
    function()


if __name__ == "__main__":
    main()
