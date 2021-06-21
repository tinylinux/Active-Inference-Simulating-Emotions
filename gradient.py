"""
Fichier contenant la fonction de calcul du gradient de l'énergie libre
"""

import numpy as np
import matools as mt

def gradF_s(s_emo, s_res, o, B_r, B_e, A, D):
    """
    Calcul du gradient de F (énergie libre) selon s
    
    Input
    ------
        s_emo : [list : vector]
            Etats internes - Emotions
        s_res : [list : vector]
            Etats internes - Ressentis
        o : [list : vector]
            Observations/ Etats physiologiques
        B_r : matrix
            Transitions entre émotions
        B_e : matrix
            Transitions entre actions
        A : matrix
            Lien Etats internes et physiologiques
        D : vector * vector
            Etats initiaux
    
    Output
    -------
        [list : vector * vector]
    """
    
    logA = mt.logA(A, s_emo, o)
    logB_r = np.log(B_r)
    logB_e = np.log(B_e)

    T = []
    n = len(s_emo)
    t = len(o)
    ke, kr = np.shape(s_emo[0]), np.shape(s_res[0])
    
    if n > 1:
        v_emo = np.ones(ke) + np.log(s_emo[0]) + np.inner(o[0], logA) + np.inner(s[1], logB_e) + np.log(D[0])
        v_res = np.ones(kr) + np.log(s_res[0]) + np.inner(s[1], logB_r) + np.log(D[1])
        T.append((v_emo, v_res))
    for i in range(1, min(n-1, t)):
        v_emo = np.ones(ke) + np.log(s_emo[i]) + np.inner(o[i], logA) + np.inner(s_emo[i+1], logB_e) + np.dot(logB_e, s_emo[i-1])
        v_res = np.ones(kr) + np.log(s_res[i]) + np.inner(s_res[i+1], logB_r) + np.dot(logB_r, s_res[i-1])
        T.append((v_emo, v_res))
    for i in range(t, n-1)
        v_emo = np.ones(ke) + np.log(s_emo[i]) + np.inner(s_emo[i+1], logB_e) + np.dot(logB_e, s_emo[i-1])
        v_res = np.ones(kr) + np.log(s_res[i]) + np.inner(s_res[i+1], logB_r) + np.dot(logB_r, s_res[i-1])
        T.append((v_emo, v_res))
    if n > 1:
        v_emo = np.ones(ke) + np.log(s_emo[n-1]) + np.dot(logB_e, s_emo[n-2])
        v_res = np.ones(kr) + np.log(s_res[n-1]) + np.dot(logB_r, s_res[n-2])
        if t >= n:
            v_emo += np.inner(o[n-1], logA)
        T.append((v_emo, v_res))
    return T


def gradF_v(s, o, B, A, D):
    """
    Calcul du gradient de l'énergie libre suivant la variable v
    
    Input
    ------
        s : [list : vector * vector]
            Etats internes
        o : [list : vector]
            Outcomes
        B : matrix * matrix
            Transition entre états
        A : matrix
            Lien entre états et outcomes
        D : vector * vector
            Vecteurs initiaux
    
    Output
    -------
        [list : vector * vector]
            Gradient selon s_emotionnel et s_ressenti
    """
    s_emo = []
    s_res = []
    for k in s:
        s_emo.append(k[0])
        s_res.append(k[1])
    
    (gse, gsr) = gradF_s(s_emo, s_res, o, B[0], B[1], A, D)
    
    L = []
    for i in range(len(s)):
        d_emo = np.diag(s_emo[i])
        d_res = np.diag(s_res[i])
        c_emo = np.inner(s_emo[i], s_emo[i])
        c_res = np.inner(s_res[i], s_res[i])
        v_emo = np.dot((d_emo - c_emo), gse[i])
        v_res = np.dot((d_res - d_res), gsr[i])
        L.append((v_emo, v_res))
    return L
