"""
   Il condizionamento dei problemi dipende dal tipo di soluzione che troviamo al problema, non dall'algoritmo che
   si implementa, soluzioni diverse hanno quindi condizionamenti diversi.
   Un problema si dice ben condizionato quando a piccole perturbazioni sui dati in ingresso corrispondono piccole
   perturbazioni dei dati in uscita.
"""


import numpy as np


class SumConditioning:

    def __init__(self, first_number, second_number, perturbation):
        self.__first_number = first_number
        self.__second_number = second_number
        self.__perturbation = perturbation
        self.__first_perturbed = self.__first_number + self.__perturbation
        self.__second_perturbed = self.__second_number + self.__perturbation
        self.__perturbed_sum = self.__first_perturbed + self.__second_perturbed
        self.__normal_sum = self.__first_number + self.__second_number

    def get_absolute_error_sum(self):
        return np.abs(self.__perturbed_sum - self.__normal_sum)

    def get_relative_error_sum(self):
        if self.__normal_sum != 0:
            return np.abs(self.__perturbed_sum - self.__normal_sum) / np.abs(self.__perturbed_sum)
        else:
            return 0

    def get_first_number_absolute_error(self):
        return np.abs(self.__first_perturbed - self.__first_number)

    def get_first_number_relative_error(self):
        if self.__first_perturbed != 0:
            return np.abs(self.__first_perturbed - self.__first_number) / np.abs(self.__first_perturbed)
        else:
            return 0

    def get_second_number_absolute_error(self):
        return np.abs(self.__second_perturbed - self.__second_number)

    def get_second_number_relative_error(self):
        if self.__second_perturbed != 0:
            return np.abs(self.__second_perturbed - self.__second_number) / self.__second_perturbed
        else:
            return 0
