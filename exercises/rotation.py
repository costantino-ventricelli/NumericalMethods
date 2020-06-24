"""
In questo esercizio applichiamo il principio delle matrici di rotazione per roteare una figura disegnata su un grafico,
la figura in questione è una casetta stilizzata,
"""
import numpy as np
from matplotlib import pyplot as plot


class Rotation:
    #   Array contenenti le coordinate dei punti che costruiranno la figura della casetta.
    __X_POINT = np.array([2, 11, 2, 11, 5, 7, 7, 1, 11])
    __Y_POINT = np.array([1, 1, 10, 10, 4, 4, 1, 14, 14])

    def __init__(self, rotation):
        self.__rotation_matrix = np.array([np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), -np.cos(rotation)])




