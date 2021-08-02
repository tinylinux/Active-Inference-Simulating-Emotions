"""
Fichier traitant de la partie minimisation de l'énergie libre
Pour le calcul de G, on s'appuie sur l'Appendix D de l'article - A synthesis
"""

import numpy as np
import mathtools as mt
import display as dp
import parameters as pm
import time

def G(s, A, C, pol, nemo=pm.N_states_emo , debug=False):
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
    Amb = -1 * mt.ambrep(s[0:nemo], s[-1*nemo:], nemo)
    H = - np.diag(mt.o_logA(A, A))
    R1 = np.inner(H, np.matrix(s))
    As = np.dot(A, s)
    R2 = mt.o_logA(As, np.transpose(np.matrix(As)))
    R3 = mt.o_logA(As, np.transpose(np.matrix(C)))
    if debug:
        print("Policy ", pol, " ", R1, " ", R2-R3)
    return Amb + R1 + R2 - R3


def get_policies(s, A, C, pol=pm.Policy, t=pm.T, nemo=pm.N_states_emo):
    """
    Compute policy at each time

    Input
    -----
        s :     vector
            Emotionnal hidden states
        A :     Dict of Matrix
            Probability of outcomes knowing states for each policy
        C :     Dict of Vector
            Probability of outcomes for each policy
        pol :   Dict of List
            Policies
        t :     Integer
            Time of emotions
        nemo :  integer
            Number of emotional states

    Output
    ------
        List of List of int
            Policies for each time
    """
    l = list(pol.keys())
    L = []
    for i in range(t):
        E = []
        for k in range(len(l) - nemo):
            E.append(G(s[k, i, 0:nemo], A[l[k]], C[l[k]][i, :], l[k]))
        L.append(mt.get_all_min(E))
    return L


def fix_s(s, A, t=pm.T, n=pm.N_states):
    """
    Create states for each time

    Input
    -----
        s : vector
            Emotionnal hidden states
        A : List of List of int
            List of policies for each time
        t : integer
            Time of experience
        n : integer
            Number of states

    Output
    ------
        Vector
            Vector for each time
    """
    f = np.zeros((t, n))
    for i in range(t):
        for k in A[i]:
            f[i, :] += s[k, i, :]
            #if i < pm.T - 1:
            #    f[i+1, pm.N_states_emo + 1 + k] = 1
        #f[i, :] = f[i, :] / len(A[i])
    return f
