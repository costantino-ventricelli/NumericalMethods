from exercises.epsilon_machine import EpsilonMachine
from exercises.real_min_max import RealMinMax
from exercises.sum_conditioning import SumConditioning
from exercises.product_conditioning import ProductConditioning
from exercises.function_conditioning import FunctionConditioning
from exercises.product_sum_test import ProductSumTest


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


def main():
    print("Inserisci il numero del problema da richiamare:\n"
          "1) Condizionamento somma;\n"
          "2) Condizionamento prodotto;\n"
          "3) Condizionamento calcolo funzioni;\n"
          "4) Epsilon machine;\n"
          "5) Minimo e massimo macchina;\n"
          "6) Somma prodotto differenza;\n")
    print("Inserisci la tua scelta: ")
    switch(int(input()))


def switch(select):
    switcher = {
        1: sum_conditioning,
        2: product_conditioning,
        3: function_conditioning,
        4: epsilon_machine_calculus,
        5: real_min_max_calculus,
        6: sum_product_test
    }
    function = switcher.get(select, lambda: "Selezione non valida")
    function()


if __name__ == "__main__":
    main()
