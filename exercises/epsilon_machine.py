# -*- coding: utf-8 -*-
"""
   L'epsilon machine è il minimo valore necessario per passare dal numero 1.0 al suo successivo,
   da questo valore si calcola la precisione della macchina e il numero di cifre significative nella conversione
   binario/decimale.
"""


import numpy as np


class EpsilonMachine:
    __STARTING_VALUE = 1.0
    __NUMERIC_BASE = 2.0

    def __init__(self):
        self.__epsilon = 1.0
        self.__precision = 0
        self.__significant_digits = 0

    #   L'algoritmo effettua delle divisioni successive tra la base del sistema numerico, finchè la somma tra 1.0
    #   e il valore calcolato ad ogni ripetizione non saranno uguali ad uno, e quindi avrò trovato nel ciclo precedente
    #   il valore dell'epsilon machine.
    def find_epsilon(self):
        real_epsilon = 0.0
        while (EpsilonMachine.__STARTING_VALUE + self.__epsilon) > 1:
            real_epsilon = self.__epsilon
            self.__epsilon = self.__epsilon / EpsilonMachine.__NUMERIC_BASE
        self.__epsilon = real_epsilon

    #   La precisione della macchina indica il numero di cifre che compongono la mantissa nalla word contenete la cifra
    #   nelle macchine attuali tutte le cifre saranno memorizzate di default a doppia precisione, quindi con 64bit,
    #   il risultato di questo calcolo dovrebbe essere 53.
    def find_precision(self):
        self.__precision = np.rint(1 - (np.log(self.__epsilon) / np.log(EpsilonMachine.__NUMERIC_BASE)))

    #   Il numero di cifre sigificative indica nel cambio di sistema numerico il numero di cifre da considerare nella
    #   conversione da binario a decimale.
    def find_significant_digits(self):
        self.__significant_digits = np.rint(self.__precision * np.log10(EpsilonMachine.__NUMERIC_BASE))

    def get_epsilon(self):
        return self.__epsilon

    def get_precision(self):
        return self.__precision

    def get_significant_digits(self):
        return self.__significant_digits
