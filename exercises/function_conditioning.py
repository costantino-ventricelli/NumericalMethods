"""
   Il condizionamento delle funzioni dipende sempre dalla funzione stessa e dal valore che deve calcolare,
   per l'esempio ho utilizzato una funzione radice quadrata e una tangente, le quali per valori bassi da calcolare
   si comportano in maniera diametralmente differente.
   La funzione radice è ben condizionata per valore di x0 bassi, mentre la funzione tangente è mal condizionata
   per valori di x0 bassi
"""


import numpy as np


class FunctionConditioning:

    def __init__(self, number, perturbation):
        self.__number = number
        self.__perturbation = perturbation
        self.__perturbed_number = self.__number + self.__perturbation
        self.__square_normal = np.sqrt(1 + self.__number)
        self.__square_perturbed = np.sqrt(1 + self.__perturbed_number)
        self.__tan_normal = np.tan(self.__number)
        self.__tan_perturbed = np.tan(self.__perturbed_number)

    def get_absolute_error_square(self):
        return np.abs(self.__square_perturbed - self.__square_normal)

    def get_relative_error_square(self):
        if self.__square_perturbed != 0:
            return np.abs(self.__square_perturbed - self.__square_normal)/np.abs(self.__square_perturbed)
        else:
            return 0

    def get_absolute_error_tan(self):
        return np.abs(self.__tan_perturbed - self.__tan_normal)

    def get_relative_error_tan(self):
        if self.__tan_perturbed != 0:
            return np.abs(self.__tan_perturbed - self.__tan_normal)/np.abs(self.__tan_perturbed)
        else:
            return 0
