"""
Fichier dédié à la récupération des outcomes
"""

import numpy as np
import parameters as pm
from scipy.special import softmax

C = np.loadtxt("matrices/C.m")

def gen_random_outcomes(t=5):
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
    L = [np.random.randint(0,2) for i in range(6)]
    V = np.zeros(pm.N_outcomes)
    for i in range(6):
        V[1+2*i] = L[i]
        V[2+2*i] = 1 - L[i]
    O = []
    for i in range(t):
        O.append(np.array(V))
    if t > 0:
        O[0][0] = 100
    for i in range(1,t):
        k = np.random.randint(0,6)
        O[i][1+2*k] = L[k]*100
        O[i][2+2*k] = (1-L[k])*100
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
