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
        for i in range(len(function), 0):
            calculated_value += x0_value**i * function[i]
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
            if iteration != 0:
                iteration = np.log2((bk[k] - ak[k]) / ZeroMethods.__tolerance)
            else:
                iteration = 1
            while k < iteration and (bk[k] - ak[k]) < ZeroMethods.__tolerance:
                ck.append((bk[k] - ak[k]) / 2)
                fc.append(ZeroMethods.get_function_value(function, ck[k]))
                check = fa[k] * fc[k]
                if check < 0:
                    ak.append(ak[k])
                    bk.append(ck[k])
                elif check > 0:
                    ak.append(ck[k])
                    bk.append(bk[k])
                else:
                    k = iteration
                k += 1
        return ck[k]

    @staticmethod
    def newton_method(function, interval):
        h = 0.1e-6
        k = 0
        xk = [ZeroMethods.bisection_method(function, interval, 0), 0]
        while np.abs(xk[k + 1] - xk[k]) > ZeroMethods.__tolerance:
            derived = (ZeroMethods.get_function_value(function, xk[k] - h) - ZeroMethods.get_function_value(function,
                                                                                                            xk[k])) / h
            xk.append(xk[k] - ZeroMethods.get_function_value(function, xk[k]) / derived)
            k += 1
        return xk[k]

    @staticmethod
    def secant_method(function, interval):
        k = 0
        xk = [ZeroMethods.bisection_method(function, interval, 0), 0]
        while np.abs(xk[k + 1] - xk[k]) > ZeroMethods.__tolerance:
            if k > 0:
                coefficient = (xk[k] - xk[k - 1]) / (ZeroMethods.get_function_value(function, xk[k]) -
                                                     ZeroMethods.get_function_value(function, xk[k - 1]))
            else:
                coefficient = xk[k] / ZeroMethods.get_function_value(function, xk[k])
            xk.append(xk[k] - ZeroMethods.get_function_value(function, xk[k]) * coefficient)
            k += 1
        return xk[k]
