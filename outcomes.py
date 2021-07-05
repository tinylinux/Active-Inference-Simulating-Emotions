"""
Fichier dédié à la récupération des outcomes
"""

import numpy as np
import parameters as pm
from scipy.special import softmax

C = np.loadtxt("matrices/C.m")

def gen_random_outcomes(t=3):
    """
    Fonction de génération des observations aléatoires

    Input
    -----
        t : int
            temps d'observation

    Output
    ------
        [list : vector]
            Listes de distribution sur les Observations
    """
    L = [np.random.randint(0,2) for i in range(5)]
    V = np.zeros(N_outcomes)
    V[-2] = 1
    for i in range(5):
        V[1+2*i] = L[i]
        V[2+2*i] = 1 - L[i]
    O = []
    for i in range(t):
        O.append(np.array(V))
    if t > 0:
        O[0][0] = 1
    return O

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
