"""
Fichier traitant de la partie minimisation de l'énergie libre
Pour le calcul de G, on s'appuie sur l'Appendix D de l'article - A synthesis
"""

import numpy as np
import mathtools as mt
import display as dp
import parameters as pm
import time

def G(s, A, C):
    """
    Compute free energy for each policy

    Input
    -----
        s : vector
            Emotionnal hidden states
        A : matrix
            Probability of observation knowing the emotion
        C : vector
            Global probability of outcomes

    Output
    ------
        Scalar
            Free energy for a policy
    """
    H = - np.diag(mt.o_logA(A, A))
    R1 = np.inner(H, np.matrix(s))
    As = np.dot(A, s)
    R2 = mt.o_logA(As, np.transpose(np.matrix(As)))
    R3 = mt.o_logA(As, np.transpose(np.matrix(C)))
    return R1 + R2 - R3


def get_policies(s, A, C):
    """
    Compute policy at each time

    Input
    -----
        s : vector
            Emotionnal hidden states
        A : Dict of Matrix
            Probability of outcomes knowing states for each policy
        C : Dict of Vector
            Probability of outcomes for each policy

    Output
    ------
        List of List of int
            Policies for each time
    """
    l = list(pm.Policy.keys())
    L = []
    for i in range(pm.T):
        E = []
        for k in range(len(l)):
            E.append(G(s[k, i, 0:pm.N_states_emo], A[l[k]], C[l[k]]))
        L.append(mt.get_all_min(E))
    return L


def fix_s(s, A):
    """
    Create states for each time

    Input
    -----
        s : vector
            Emotionnal hidden states
        A : List of List of int
            List of policies for each time

    Output
    ------
        Vector
            Vector for each time
    """
    f = np.zeros((pm.T, pm.N_states))
    f[0, pm.N_states_emo] = 1
    for i in range(pm.T):
        for k in A[i]:
            f[i, 0:pm.N_states_emo] += s[k, i, 0:pm.N_states_emo]
            if i < pm.T - 1:
                f[i+1, pm.N_states_emo + 1 + k] = 1
        f[i, :] = f[i, :] / len(A[i])
    return f
