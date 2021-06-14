"""
Fichier contenant la fonction de calcul du gradient de l'énergie libre
"""

import numpy as np
import matools as mt

def gradF_s(s, o, B, A, D):
    """
    Calcul du gradient de F (énergie libre) selon s
    
    Input
    ------
        s : [list : vector]
            Etats internes
        o : [list : vector]
            Observations/ Etats physiologiques
        B : matrix
            Transitions entre états
        A : matrix
            Lien Etats internes et physiologiques
        D : vector
            Etats initiaux
    
    Output
    -------
        [list : vector]
    """
    
    logA = mt.logA(A, s, o)
    logB = np.log(B)

    T = []
    n = len(s)
    t = len(o)
    k = np.shape(s)
    
    if n > 1:
        v = np.ones(k) + np.log(s[0]) + np.inner(o[0], logA) + np.inner(s[1], logB) + np.log(D)
        T.append(v)
    for i in range(1, min(n-1, t)):
        v = np.ones(k) + np.log(s[i]) + np.inner(o[i], logA) + np.inner(s[i+1], logB) + np.dot(logB, s[i-1])
        T.append(v)
    for i in range(t, n-1)
        v = np.ones(k) + np.log(s[i]) + np.inner(s[i+1], logB) + np.dot(logB, s[i-1])
        T.append(v)
    if n > 1:
        v = np.ones(k) + np.log(s[n-1]) + np.dot(logB, s[n-2])
        if t >= n:
            v += np.inner(o[n-1], logA)
        T.append(v)
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
