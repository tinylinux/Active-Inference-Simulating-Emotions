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


def observ_gen(t=pm.T):
    """
    Generate observations for simulation

    Input
    ------
        t:  integer
            Time of observations

    Output
    ------
        O:  matrix
            Matrix of observations
    """
    L = [np.random.randint(1, pm.N_outcomes) for i in range(t)]
    O = np.zeros((t, pm.N_outcomes))
    for i in range(t):
        O[i, L[i]] = 1
    return O

def get_outcomes():
    """
    Load observations generated

    Output
    ------
        out:    matrix
            Matrix with all observations
    """
    # To define it clearly - here it is an example
    out = np.matrix(np.loadtxt("matrices/outcomes/o.t"))
    return out
