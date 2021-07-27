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
            Gradient of free energy according to emotionnal states
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
    for i in range(T, tau-1):
        g[i, :] = 1 + np.log(s_emo[i, :]) - mt.logB_s(B_emo,s_emo[i-1, :]) - mt.o_logA(s_emo[i+1, :], B_emo)
    if tau > T:
        g[tau-1, :] = 1 + np.log(s_emo[tau-1, :]) - mt.logB_s(B_emo,s_emo[tau-2, :])
    else:
        g[tau-1, :] = 1 + np.log(s_emo[tau-1, :]) - mt.o_logA(o[tau-1, :], A) - mt.logB_s(B_emo,s_emo[tau-2, :])
    return g


def gradF_sp(s_emo, s_p, o, Ap, D, B, tau, T):
    """
    Gradiant according to attentional focus states

    Input
    -----
        s_emo : matrix
            emotionnal states
        s_p :   matrix
            attentional focus states
        o :     matrix
            observations
        Ap :    matrix
            Probability of state knowing the observations
        B :     matrix
            Probability of state knowing the former one
        tau :   integer
            Time of states
        T :     integer
            Time of observations

    Output
    ------
        matrix
            Gradient of free energy according to attentional states
    """
    g = np.matrix(np.zeros(np.shape(s_p)))

    if tau <= 0:
        return g
    else:
        if T <= 0:
            g[0, :] = 1 + np.log(s_p[0, :]) - np.log(D) - mt.o_logA(s_p[1, :], B)
        else:
            A_rep = mt.construct_A_report(s_emo[0, :], pm.N_states_emo)
            oAf = mt.o_logA(o[0, :-2], Ap)
            oAr = mt.o_logA(o[0, -2:], A_rep)
            oA = np.concatenate((oAf, oAr), axis=1)
            g[0, :] = 1 + np.log(s_p[0, :]) - oA - np.log(D) - mt.o_logA(s_p[1, :], B)
    for i in range(1, min(T, tau-1)):
        A_rep = mt.construct_A_report(s_emo[i, :], pm.N_states_emo)
        oAf = mt.o_logA(o[i, :-2], Ap)
        print(A_rep)
        oAr = mt.o_logA(o[i, -2:], A_rep)
        oA = np.concatenate((oAf, oAr), axis=1)
        g[i, :] = 1 + np.log(s_p[i, :]) - oA - mt.o_logA(s_p[i+1, :], B) - mt.logB_s(B,s_p[i-1, :])
    for i in range(T, tau-1):
        g[i, :] = 1 + np.log(s_p[i, :]) - mt.logB_s(B,s_p[i-1, :]) - mt.o_logA(s_p[i+1, :], B)
    if tau > T:
        g[tau-1, :] = 1 + np.log(s_p[tau-1, :]) - mt.logB_s(B,s_p[tau-2, :])
    else:
        A_rep = mt.construct_A_report(s_emo[0, :], pm.N_states_emo)
        oAf = mt.o_logA(o[0, :-2], Ap)
        print(A_rep)
        oAr = mt.o_logA(o[0, -2:], A_rep)
        oA = np.concatenate((oAf, oAr), axis=1)
        g[tau-1, :] = 1 + np.log(s_p[tau-1, :]) - oA - mt.logB_s(B,s_p[tau-2, :])
    return g

def gradF_v(s_emo, s_p, oE, oP, A, Ap, D, B_emo, B_act, tau, T=pm.T):
    """
    Calcul du gradient de l'énergie libre suivant la variable v

    Input
    ------
        s_emo : matrix
            Emotionnal states for each time
        s_p : matrix
            Emotionnal states for each time
        oE :     matrix
            Observations for each time for emotions
        oP :     matrix
            Observations for each time for policies
        A :     matrix
            Probability of state knowing the observations
        Ap :    matrix
            Probability of state knowing the observations for policies
        D :     vector
            Initial state probability
        B_emo : matrix
            Probability of future state knowing the former state
        B_act : matrix
            Probability of future state knowing the former state for policies
        tau :   integer
            Time of states
        T :     integer
            Time of observation

    Output
    -------
        [list : vector * vector]
            Gradient selon s_emotionnel et s_ressenti
    """
    f = gradF_s(s_emo, oE, A, D[0:pm.N_states_emo], B_emo, tau, T)
    g = gradF_sp(s_emo, s_p, oP, Ap, D[pm.N_states_emo:], B_act, tau, T)

    for i in range(tau):
        d_emo = np.diag(s_emo[i, :])
        d_p = np.diag(s_p[i, :])
        c_emo = np.inner(np.transpose(np.matrix(s_emo[i])), np.transpose(np.matrix(s_emo[i])))
        c_p = np.inner(np.transpose(np.matrix(s_p[i])), np.transpose(np.matrix(s_p[i])))
        v_emo = np.dot((d_emo - c_emo), np.transpose(f[i]))
        v_p = np.dot((d_p - c_p), np.transpose(g[i]))
        f[i, :] = np.transpose(v_emo)
        g[i, :] = np.transpose(v_p)
    return (f, g)
