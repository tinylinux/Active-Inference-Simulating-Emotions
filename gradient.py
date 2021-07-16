"""
Fichier contenant la fonction de calcul du gradient de l'énergie libre
"""

import numpy as np
import mathtools as mt
import parameters as pm

def gradF_s(s_emo, o, A, D, B_emo, tau, T):
    """
    Calcul du gradient

    Input
    -----
        s_emo : matrix
            Emotionnal states for each time
        o :     matrix
            Observations for each time
        A :     matrix
            Probability of state knowing the observations
        D :     vector
            Initial state probability
        B_emo : matrix
            Probability of future state knowing the former one
        tau :   integer
            Time of states
        T :     integer
            Time of observation

    Output
    ------
        matrix
            Gradient of free energy according states
    """
    g = np.matrix(np.zeros(np.shape(s_emo)))

    if tau <= 0:
        return g
    else:
        if T <= 0:
            g[0, :] = 1 + np.log(s_emo[0, :]) - np.log(D) - mt.o_logA(s_emo[1, :], B_emo)
        else:
            g[0, :] = 1 + np.log(s_emo[0, :]) - mt.o_logA(o[0, :], A) - np.log(D) - mt.o_logA(s_emo[1, :], B_emo)
    for i in range(1, min(T, tau-1)):
        g[i, :] = 1 + np.log(s_emo[i, :]) - mt.o_logA(o[i, :], A) - mt.o_logA(s_emo[i+1, :], B_emo) - mt.logB_s(B_emo,s_emo[i-1, :])
    for i in range(tau-1, T):
        g[i, :] = 1 + np.log(s_emo[i, :]) - mt.logB_s(B_emo,s_emo[i-1, :])
    if tau > T:
        g[tau-1, :] = 1 + np.log(s_emo[tau-1, :]) - mt.logB_s(B_emo,s_emo[tau-2, :])
    else:
        g[tau-1, :] = 1 + np.log(s_emo[tau-1, :]) - mt.o_logA(o[tau-1, :], A) - mt.logB_s(B_emo,s_emo[tau-2, :])
    return g


def gradF_v(s, o, A, D, B_emo, tau, T=pm.T):
    """
    Calcul du gradient de l'énergie libre suivant la variable v

    Input
    ------
        s_emo : matrix
            Emotionnal states for each time
        o :     matrix
            Observations for each time
        A :     matrix
            Probability of state knowing the observations
        D :     vector
            Initial state probability
        B_emo : matrix
            Probability of future state knowing the former state
        tau :   integer
            Time of states
        T :     integer
            Time of observation

    Output
    -------
        [list : vector * vector]
            Gradient selon s_emotionnel et s_ressenti
    """
    f = gradF_s(s, o, A, D, B_emo, tau, T)

    for i in range(T):
        d_emo = np.diag(s[i, :])
        c_emo = np.inner(np.transpose(np.matrix(s[i])), np.transpose(np.matrix(s[i])))
        v_emo = np.dot((d_emo - c_emo), np.transpose(f[i]))
        np.shape(v_emo)
        f[i, :] = np.transpose(v_emo)

    return f
