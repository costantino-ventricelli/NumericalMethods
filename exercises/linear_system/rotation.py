# -*- coding: utf-8 -*-
"""
    In questo esercizio applichiamo il principio delle matrici di rotazione per roteare una figura disegnata su un
    grafico, la figura in questione è una casetta stilizzata.
    Per la rotazione della figura viene effettuata una moltiplicazione punto punto tra la matrice singolare di rotazione
    e le coordinate dei punti nel piano cartesiano.
"""
import numpy as np
from matplotlib import pyplot as plot


class Rotation:
    #   Array contenenti le coordinate dei punti che costruiranno la figura della casetta.
    __X_POINT = np.array([2, 2, 1, 5.5, 10, 9, 9, 4, 4, 6, 6, 2])
    __Y_POINT = np.array([1, 7, 6, 10, 6, 7, 1, 1, 4, 4, 1, 1])
    #   Assegno i nomi ai valori 0 e 1 per maggior chiarezza.
    __X = int(0)
    __Y = int(1)

    def __init__(self, rotation):
        #   Trasforma i gradi in radianti.
        self.__rotation_radians = np.radians(rotation)
        #   Questa è la matrice unitaria di rotazione.
        self.__rotation_matrix = np.array([[np.cos(self.__rotation_radians), np.sin(self.__rotation_radians)],
                                           [-np.sin(self.__rotation_radians), np.cos(self.__rotation_radians)]])
        #  Questi due comandi definiscono i limiti del piano cartesiano.
        plot.xlim(-15, 15)
        plot.ylim(-15, 15)
        plot.plot(Rotation.__X_POINT, Rotation.__Y_POINT)

    """
        Rotate scandice l'unione dei vettori __X_POINT e __Y_POINT di ogni coppia ne effettua la moltiplicazione punto
        punto con la matrice di rotazione e ne memorizza i valori in due nuovi vettori: rotate_x e rotate_y, dopo aver 
        scansionato tutta la "matrice" iterabile creata con il comando zip(), la nuova figura viene disegnata grazie ai 
        due vettori generati e mostrata sul grafico.
    """
    def rotate(self):
        rotate_x = []
        rotate_y = []
        for x, j in zip(Rotation.__X_POINT, Rotation.__Y_POINT):
            initial_point = np.array([x, j])
            rotate_point = np.dot(self.__rotation_matrix, initial_point)
            rotate_x.append(rotate_point[self.__X])
            rotate_y.append(rotate_point[self.__Y])
        plot.plot(rotate_x, rotate_y)
        plot.show()

