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
        for i in range(n):
            nodes[i] = (0.5 * (start_interval + end_interval)) + 0.5 * (start_interval - end_interval) \
                       * np.cos(i/n * np.pi)
        return nodes
