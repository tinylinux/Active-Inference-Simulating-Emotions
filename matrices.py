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
        D:      vector
            Initial states
        A:      matrix dict
            A matrix (Observation/State) for each action
        B:      matrix dict
            B matrix (States transitions) for each action
        B_emo:  matrix
            B matrix (States transitions) for emotions
        C:      vector
            probability of observation
    """
    D = np.loadtxt("matrices/D.txt")
    A = {}
    B = {}
    C = {}
    C_imp = softmax(np.loadtxt("matrices/C.txt"))
    for k in pm.Policy:
        A[k] = np.matrix(np.loadtxt("matrices/A_" + str(k) + ".txt"))
        A[k] = A[k][:, 2:4]
        B[k] = np.matrix(np.loadtxt("matrices/B_" + str(k) + ".txt"))
        s = len(pm.Policy[k])
        c = np.ones(s)
        for i in range(s):
            c[i] = C_imp[pm.Policy[k][i]]
        C[k] = np.array(c)
    B_emo = np.matrix(np.loadtxt("matrices/B_emo.txt"))
    return (D, A, B, B_emo, C)


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
