"""
Fichier pour importer les matrices
"""

import numpy as np
from scipy.special import softmax
import parameters as pm

def import_matrices():
    """
    Import all matrices necessary for simulation

    Output
    ------
        D:  vector
            Initial states
        A:  matrix dict
            A matrix (Observation/State) for each action
        B:  matrix dict
            B matrix (States transitions) for each action
        C:  vector
            probability of observation
    """
    D = np.loadtxt("matrices/D.txt")
    A = {}
    B = {}
    for k in pm.Policy:
        A[k] = np.matrix(np.loadtxt("matrices/A_" + str(k) + ".txt"))
        B[k] = np.matrix(np.loadtxt("matrices/B_" + str(k) + ".txt"))
    C = np.loadtxt("matrices/C.txt")
    return (D, A, B, C)
