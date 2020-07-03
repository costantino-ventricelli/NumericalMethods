# -*- coding: utf-8 -*-
"""
   In questa classe verranno effettuati due test in tre modalità differenti:
   1) Dato un numero macchina calcolare la differenza di ruisultati tra una serie di somme ed un prodotto con numeri
       in doppia, singola e mezza precisione;
   2) Dato un numero non di macchina calcolare le stesse differenze con le tre diverse precisioni.
   Risultato più alto è il numero per cui effettuare la moltiplicazione o ripetere le somme e maggiori sono differenze
   tra la serie di somme e il prodotto.
"""


import numpy as np


class ProductSumTest:

    # La generazione pseudo random dovrebbe dare come risultato un numero macchina direttamtente.
    __MACHINE_NUMBER = np.random.rand()
    # Numero reale scelto a caso
    __REAL_NUMBER = 6.92

    def __init__(self, times):
        self.__times = int(times)
        self.__machine_sum_double = np.float64(0.00)
        self.__machine_sum_single = np.float32(0.00)
        self.__machine_sum_half = np.float16(0.00)
        self.__real_sum_double = np.float64(0.00)
        self.__real_sum_single = np.float32(0.00)
        self.__real_sum_half = np.float16(0.00)

    def calculate_machine_products_sum(self):
        for i in range(0, self.__times):
            self.__machine_sum_double = np.float64(self.__machine_sum_double + ProductSumTest.__MACHINE_NUMBER)
            self.__machine_sum_single = np.float32(self.__machine_sum_single + ProductSumTest.__MACHINE_NUMBER)
            self.__machine_sum_half = np.float16(self.__machine_sum_half + ProductSumTest.__MACHINE_NUMBER)

    def calculate_real_products_sum(self):
        for i in range(0, self.__times):
            self.__real_sum_double = np.float64(self.__real_sum_double + ProductSumTest.__REAL_NUMBER)
            self.__real_sum_single = np.float32(self.__real_sum_single + ProductSumTest.__REAL_NUMBER)
            self.__real_sum_half = np.float16(self.__real_sum_half + ProductSumTest.__REAL_NUMBER)

    def get_machine_double_sum(self):
        return self.__machine_sum_double

    def get_machine_single_sum(self):
        return self.__machine_sum_single

    def get_machine_half_sum(self):
        return self.__machine_sum_half

    def get_machine_double_product(self):
        return np.float64(ProductSumTest.__MACHINE_NUMBER * self.__times)

    def get_machine_single_product(self):
        return np.float32(ProductSumTest.__MACHINE_NUMBER * self.__times)

    def get_machine_half_product(self):
        return np.float16(ProductSumTest.__MACHINE_NUMBER * self.__times)

    def get_real_double_sum(self):
        return self.__real_sum_double

    def get_real_single_sum(self):
        return self.__real_sum_single

    def get_real_half_sum(self):
        return self.__real_sum_half

    def get_real_double_product(self):
        return np.float64(np.float64(ProductSumTest.__REAL_NUMBER) * self.__times)

    def get_real_single_product(self):
        return np.float32(np.float32(ProductSumTest.__REAL_NUMBER) * self.__times)

    def get_real_half_product(self):
        return np.float16(np.float16(ProductSumTest.__REAL_NUMBER) * self.__times)






