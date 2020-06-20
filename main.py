from exercises.epsilon_machine import EpsilonMachine
from exercises.real_min_max import RealMinMax
from exercises.sum_conditioning import SumConditioning
from exercises.product_conditioning import ProductConditioning


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
    print("Errore assoluto somma: %15.12f" %problem.get_absolute_error_sum())
    print("Errore relativo somma: %15.12f" %problem.get_relative_error_sum())
    print("Errore assoluto primo numero: %15.12f" %problem.get_first_number_absolute_error())
    print("Errore relativo primo numero: %15.12f" %problem.get_second_number_relative_error())
    print("Errore assoluto secondo numero: %15.12f" %problem.get_second_number_absolute_error())
    print("Errore relativo secondo numero: %15.12f" %problem.get_second_number_relative_error())


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
    return 0


def main():
    print("Inserisci il numero del problema da richiamare:\n"
          "1) Condizionamento somma;\n"
          "2) Condizionamento prodotto;\n"
          "3) Condizionamento calcolo funzioni;\n"
          "4) Epsilon machine;\n"
          "5) Minimo e massimo macchina;\n")
    print("Inserisci la tua scelta: ")
    switch(int(input()))


def switch(select):
    switcher = {
        1: sum_conditioning,
        2: product_conditioning,
        3: function_conditioning,
        4: epsilon_machine_calculus,
        5: real_min_max_calculus
    }
    function = switcher.get(select, lambda: "Selezione non valida")
    function()


if __name__ == "__main__":
    main()
