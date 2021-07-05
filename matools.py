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

class Shape(Exception):
    pass

def inner(X, Y):
    """
    Le produit inner, défini comme X.Y = X^T Y

    Input
    ------
        X : matrix
        Y : matrix

    Output
    -------
        matrix
    """
    A = np.matrix(X)
    B = np.matrix(Y)
    (a1, a2) = np.shape(A)
    (b1, b2) = np.shape(B)
    if (a1 == b1):
        return np.dot(np.transpose(A), B)
    elif (a2 == b1):
        return np.dot(A, B)
    else:
        try:
            raise Shape()
        except Shape:
            print("problème de dimension A ", (a1,a2), " et B ", (b1,b2))

def norm(A):
    """
    Calcul de la norme infinie

    Input
    -----
        A : matrix
            Matrice

    Output
    ------
            scalar
    """
    D = [np.max(np.abs(k)) for k in A]
    return np.max(D)

def addmatvec(A, M):
    """
    Additionner une matrice colonne et un vecteur

    Input
    -----
        M : matrix
        A : vector

    Output
    ------
        vector
    """
    k = np.shape(M)
    v = np.array(A)
    if k[0] == 1:
        a = k[1]
        for i in range(a):
            v[i] += M[0, i]
    else:
        a = k[0]
        for i in range(a):
            v[i] += M[i, 0]
    return v

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

def sub3(a, b, c):
    """
    Opération infinie a + b - c
    """
    if a == np.inf and c == np.inf:
        return b
    elif a == - np.inf and c == - np.inf:
        return b
    elif b == np.inf and c == np.inf:
        return a
    elif b == - np.inf and c == - np.inf:
        return a
    else:
        return a + b - c

def subinf(A, B):
    """
    Soustraction entre A et B suivant si on a l'infini

    Input
    -----
        A : array_like
        B : array_like

    Output
    ------
        array_like
    """
    if A.ndim == 1:
        k = A.shape
        C = np.zeros(k)
        k1 = A.size
        for i in range(k1):
            if A[i] == np.inf and B[i] == np.inf:
                C[i] = 0
            elif A[i] == -np.inf and B[i] == -np.inf:
                C[i] = 0
            else:
                C[i] = A[i] - B[i]
        return C
    if A.ndim == 2:
        (k1, k2) = A.shape
        C = np.zeros((k1, k2))
        for i in range(k1):
            for j in range(k2):
                if A[i,j] == np.inf and B[i,j] == np.inf:
                    C[i,j] = 0
                else:
                    C[i,j] = A[i,j] - B[i,j]
        return C

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
        t = np.kron(o[i], s[i]).reshape((13,4))
        a += t
        a_0 += t
    return (psi(a) - psi(a_0))
