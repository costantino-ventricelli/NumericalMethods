# -*- coding: utf-8 -*-
import time

import numpy as np

from exercises.epsilon_machine import EpsilonMachine
from exercises.function_conditioning import FunctionConditioning
from exercises.linear_system_solver import LinearTest
from exercises.product_conditioning import ProductConditioning
from exercises.product_sum_test import ProductSumTest
from exercises.real_min_max import RealMinMax
from exercises.rotation import Rotation
from exercises.substitution_algorithm import SubstitutionAlgorithm
from exercises.sum_conditioning import SumConditioning
from exercises.vandermonde_matrix import VandermondeMatrix


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
          product_sum_test.get_machine_double_sum(), " mentre il prodotto è: ", product_sum_test.get_machine_double_product())
    print("Il valore delle somme in doppia precisione per il numero reale è: ",
          product_sum_test.get_real_double_sum(), " mentre il prodotto è: ", product_sum_test.get_real_double_product(), "\n")
    print("Il valore delle somme in singola precisione per il numero macchina è: ",
          product_sum_test.get_machine_single_sum(), " mentre il prodotto è: ", product_sum_test.get_machine_single_product())
    print("Il valore delle somme in singola precisione per il numero reale è: ",
          product_sum_test.get_real_single_sum(), " mentre il prodotto è: ", product_sum_test.get_real_single_product(), "\n")
    print("Il valore delle somme in mezza precisione per il numero macchina è: ",
          product_sum_test.get_machine_half_sum(), " mentre il prodotto è: ", product_sum_test.get_machine_half_product())
    print("Il valore delle somme in mezza precisione per il numero reale è: ",
          product_sum_test.get_real_half_sum(), "mentre il prodotto è: ", product_sum_test.get_real_half_product(), "\n")


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


def linear_system_test():
    print("VANDERMONDE MATRIX CON METODO DI GAUSS")
    start_time = time.time()
    solution, absolute, relative = LinearTest.std_vandermonde_matrix_gauss()
    total_time = (time.time() - start_time)
    print("Tempo di esecuzione: ", total_time)
    print("Soluzione: ", solution, "\nErrore assoluto: ", absolute, "\nRelative: ", relative, "\n")
    print("HILBERT MATRIX CON METODO DI GAUSS")
    start_time = time.time()
    solution, absolute, relative = LinearTest.std_hilbert_matrix_gauss()
    total_time = (time.time() - start_time)
    print("Tempo di esecuzione: ", total_time)
    print("Soluzione: ", solution, "\nErrore assoluto: ", absolute, "\nRelative: ", relative, "\n")
    print("VANDERMONDE MATRIX CON FATTORIZZAZIONE LU")
    start_time = time.time()
    solution, absolute, relative = LinearTest.std_vandermonde_matrix_lu()
    total_time = (time.time() - start_time)
    print("Tempo di esecuzione: ", total_time)
    print("Soluzione: ", solution, "\nErrore assoluto: ", absolute, "\nRelative: ", relative, "\n")
    print("HILBERT MATRIX CON FATTORIZZAZIONE LU")
    start_time = time.time()
    solution, absolute, relative = LinearTest.std_hilbert_matrix_lu()
    total_time = (time.time() - start_time)
    print("Tempo di esecuzione: ", total_time)
    print("Soluzione: ", solution, "\nErrore assoluto: ", absolute, "\nRelative: ", relative, "\n")
    print("METODO ITERATIVO DI JACOBI")
    start_time = time.time()
    solution, absolute, relative = LinearTest.std_jacobi()
    total_time = time.time() - start_time
    print("Tempo di esecuzione: ", total_time)
    print("Soluzione: ", solution, "\nErrore assoluto: ", absolute, "\nRelative: ", relative, "\n")
    print("METODO ITERATIVO DI GAUSS-SEIDEL")
    start_time = time.time()
    solution, absolute, relative = LinearTest.std_gauss_seidel()
    total_time = time.time() - start_time
    print("Tempo di esecuzione: ", total_time)
    print("Soluzione: ", solution, "\nErrore assoluto: ", absolute, "\nRelative: ", relative, "\n")


def random_system_solver():
    print("Work in progress")


def linear_system():
    choice = int(input("Vuoi:\n"
          "1) Avviare il test standard;\n"
          "2) Avviare il test con valori casuali;\n"
          "Inserisci risposta: "))
    if choice == 1:
        linear_system_test()
    else:
        random_system_solver()


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
          "10) Sistemi lineari\n")
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
        10: linear_system
    }
    function = switcher.get(select, lambda: "Selezione non valida")
    function()


if __name__ == "__main__":
    main()
