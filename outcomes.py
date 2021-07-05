"""
Fichier dédié à la récupération des outcomes
"""

import numpy as np
import parameters as pm
from scipy.special import softmax

C = np.loadtxt("matrices/C.m")

def get_outcomes():
    """
    Fonction de récupération des observations
    pour la simulation des émotions.

    Input
    -----
        matrice

    Output
    ------
        [list : vector]
            Listes de distribution sur les observations
    """
    # To define it clearly - here it is an example
    out = []
    for i in range(3):
        out.append(np.loadtxt("matrices/outcomes/" + str(i+1) + ".t"))
    return out
