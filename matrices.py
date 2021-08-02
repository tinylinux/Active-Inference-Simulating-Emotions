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
        Af:     matrix
            A matrix (O/S) for focus (non report) states
        B:      matrix dict
            B matrix (States transitions) for each action
        B_emo:  matrix
            B matrix (States transitions) for emotions
        C:      vector
            probability of observation
    """
    D = np.loadtxt("matrices/D.txt")
    Af = np.loadtxt("matrices/Af.txt")
    A = {}
    B = {}
    C = {}
    C_imp = softmax(np.loadtxt("matrices/C.txt"))
    for k in pm.Policy:
        A[k] = np.matrix(np.loadtxt("matrices/A_" + str(k) + ".txt"))
        A[k] = A[k][:, 0:4:3]
        B[k] = np.matrix(np.loadtxt("matrices/B_" + str(k) + ".txt"))
        s = len(pm.Policy[k])
        c = np.ones(s)
        for i in range(s):
            c[i] = C_imp[pm.Policy[k][i]]
        C[k] = np.array(c)
    B_emo = np.matrix(np.loadtxt("matrices/B_emo.txt"))
    return (D, A, Af, B, B_emo, C)


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
        Og: matrix
            Matrix of global observations
    """
    L = [np.random.randint(1, pm.N_outcomes) for i in range(t)]
    O = np.zeros((t, pm.N_outcomes))
    O[0, 0] = 1
    O[0, 5] = 1
    for i in range(1,3):
        O[i, 3] = 1
    for i in range(3, t):
        O[i, 2] = 1
    for i in range(1, t):
        O[i, 5] = 1
    D = (pm.N_outcomes - 3)//2 + 3
    Og = np.zeros((t, D))
    Og[:, 0] = O[:, 0]
    for i in range(D-3):
        Og[:, 1+i] = O[:, i*2 + 1] + O[:, i*2+2]
    Og[:, -1] = O[:, -1]
    Og[:, -2] = O[:, -2]
    return (O, Og)

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
