"""
Module contenant les outils mathématiques qui vont être utilisés
"""

import numpy as np
import scipy as sp

from scipy.special import softmax
from scipy.special import digamma
from scipy.special import gamma

log = np.log
gamma = gamma
psi = digamma
softmax = softmax

block_diag = sp.linalg.block_diag

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
        matrix
    """
    T = min(len(s), len(o))
    a = np.matrix(A)
    a_0 = a0(A)
    for i in range(T):
        t = np.kron(o[i], s[i])
        a += t
        a_0 += t
    return (psi(a) - psi(a_0))

