# -*- coding: utf-8 -*-
"""
    In questo modulo implementiamo il polinomio di chebyshev e lo applichiamo all'interpolazione di Lagrange per valutare
    quanto l'approssimazione migliori rispetto all'utilizizzo di una selezione di punti lineare.
"""

import numpy as np


class ChebyshevNodes:

    @staticmethod
    def get_chebyshev_nodes(start_interval, end_interval, n):
        nodes = np.zeros([n])
        add = 0.5 * (start_interval + end_interval)
        sub = 0.5 * (end_interval - start_interval)
        for i in range(n):
            cos_val = np.cos(((2 * i + 1)/(2 * n + 2)) * np.pi)
            nodes[i] = add + (sub * cos_val)
        return np.flip(nodes)
