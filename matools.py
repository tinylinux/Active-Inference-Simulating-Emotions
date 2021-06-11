"""
Module contenant les outils mathématiques qui vont être utilisés
"""

import numpy as np
import scipy as sp


log = np.log
gamma = sp.special.gamma
psi = sp.special.digamma

def a0(A):
    """
    Calcul de la matrice homogène de A
    ce qui va nous servir pour la partie learning - log A
    
    Input 
    ------
        A : np.matrix
    
    Output
    ------
        S : np.matrix
    """
    S = np.matrix(A)
    (n,m) = S.shape
    for i in range(n):
        s = 0
        for j in range(m):
            s += A[i,j]
        S[i,:] = s * np.ones(m)
    return S

def logA(A, s, o):
    """
    Calcul du log A d'après les articles d'AI
    
    Input
    ------
        A : matrix
        s : [list : vector]
        o : [list : vector]
    
    Output
    -------
        L : matrix
    """
    T = min(len(s), len(o))
    L = np.matrix(A)
    for i in range(T):
        L += np.kron(o[i], s[i])
    return L
