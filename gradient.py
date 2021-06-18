"""
Fichier contenant la fonction de calcul du gradient de l'énergie libre
"""

import numpy as np
import matools as mt

def gradF_s(s_emo, s_res, o, B, A, D):
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
        B : matrix * matrix
            Transitions entre états
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
    """
    gs = gradF_s(s, o, B, A, D)
    L = []
    for i in range(len(s)):
        d = np.diag(s[i])
        c = np.inner(s[i], s[i])
        v = np.dot((d - c), gs[i])
        L.append(v)
    return L
