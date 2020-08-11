# -*- coding: utf-8 -*-
"""
    In questo modulo verranno implemantati i tre metodi di risoluzione di equazioni non lineari:
    - Metodo di bisezione
    - Metodo di Newton
    - Metodo delle secanti
    Verrà poi generata una tabella per il confronto dei tre metodi risolutivi.

    functions: contiene una lista di liste le quali contengono a loro volta i coefficienti dei valori della x il cui
        grado corrisponde alla posizione nella lista:
        [1, 3, 2] = 2*x^2 + 3*x + 1.
    definition_interval: contiene per ogni funzione il suo corrispondente intervallo in cui cercare lo zero della.
"""

import numpy as np


class ZeroMethods:
    __tolerance = 1.0e-10

    """
        Questo metodo permette di calcolare il valore di una funzione in un punto, quindi function contiene la lista dei
        coefficienti secondo la formattazione precedente, metre value contiene il valore per cui si vuole calcolare la 
        funzione, in pratica l'x0.
    """

    @staticmethod
    def get_function_value(function, x0_value):
        calculated_value = 0.00
        iteration = len(function)
        for i in range(iteration):
            calculated_value += x0_value ** (iteration - 1 - i) * function[i]
        return calculated_value

    @staticmethod
    def bisection_method(function, interval, iteration):
        ak = [interval[0]]
        bk = [interval[1]]
        fa = [ZeroMethods.get_function_value(function, interval[0])]
        fc = []
        ck = []
        k = 0
        if fa[0] * ZeroMethods.get_function_value(function, interval[1]) < 0:
            if iteration == 0:
                iteration = int(np.log2((bk[k] - ak[k]) / ZeroMethods.__tolerance))
            else:
                iteration = 1
            while k < iteration and (bk[k] - ak[k]) > ZeroMethods.__tolerance:
                ck.append((bk[k] + ak[k]) / 2)
                fc.append(ZeroMethods.get_function_value(function, ck[k]))
                if fa[k] * fc[k] < 0:
                    ak.append(ak[k])
                    bk.append(ck[k])
                else:
                    ak.append(ck[k])
                    bk.append(bk[k])
                k += 1
                fa.append(ZeroMethods.get_function_value(function, ak[k]))
        return ck[k - 1]

    @staticmethod
    def newton_method(function, interval):
        h = 1.0e-11
        k = 0
        xk = [ZeroMethods.bisection_method(function, interval, 1)]
        prev = 0.00
        while np.abs(prev - xk[k]) > ZeroMethods.__tolerance:
            derivative = (ZeroMethods.get_function_value(function, xk[k] + h)
                          - ZeroMethods.get_function_value(function, xk[k])) / h
            xk.append(xk[k] - ZeroMethods.get_function_value(function, xk[k]) / derivative)
            prev = xk[k]
            k += 1
        return xk[k]

    @staticmethod
    def secant_method(function, interval):
        xk = [ZeroMethods.get_function_value(function, interval[0]), ZeroMethods.get_function_value(function,
                                                                                                    interval[1])]
        k = 1
        while (np.abs(xk[k] - xk[k - 1])) > ZeroMethods.__tolerance:
            f_k = ZeroMethods.get_function_value(function, xk[k])
            f_k_1 = ZeroMethods.get_function_value(function, xk[k - 1])
            xk.append(xk[k] - f_k * ((xk[k] - xk[k - 1]) / (f_k - f_k_1)))
            k += 1
        return xk[k]
