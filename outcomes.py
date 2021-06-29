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

    Output
    ------
        [list : vector]
            Listes de distribution sur les observations
    """
    # To define it clearly - here it is an example
    return [np.array(C) for i in range(3)]
