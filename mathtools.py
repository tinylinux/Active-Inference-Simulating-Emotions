"""
Fichier de module mathématiques
"""

import numpy as np
import scipy as sp

from numpy.linalg import norm as norm
from scipy.special import softmax as ssoftmax
from scipy.special import digamma
from scipy.special import gamma

log = np.log
gamma = gamma
psi = digamma

block_diag = sp.linalg.block_diag

class Shape(Exception):
    pass

def softmax(T):
    S = np.array(T)
    if np.inf in T:
        for i in range(len(S)):
            if T[i] == np.inf:
                S[i] = 1.0
            else:
                S[i] = - np.inf
        return ssoftmax(S)
    else:
        return ssoftmax(T)

def o_logA(o, A, debug=False):
    """
    Fonction prenant compte de l'infini calculant
    oT.log(A)

    Input
    ------
        o : vecteur
            Vector o
        A : matrix
            Matrix A

    Output
    ------
        S : matrix
            Matrix O.log(A)
    """
    O = np.matrix(o)
    A = np.matrix(np.log(A))
    if debug:
        print(A)
        print(o)
    (k1, k2) = np.shape(O)
    (k3, k4) = np.shape(A)
    if debug:
        print(np.shape(O), " ", np.shape(A))
    if k2 == k3:
        S = np.zeros((k1,k4))
        for i in range(k1):
            for j in range(k4):
                for k in range(k2):
                    if np.log(O[i,k]) > - np.inf :
                        S[i,j] += O[i,k] *  A[k, j]
        return np.matrix(S)
    else:
        try:
            Shape()
        except Shape:
            print("Problème de dimension : ", (k1, k2), " avec ", (k3,k4))
            exit()


def logB_s(B, s):
    """
    Fonction prenant compte de l'infini calculant
    log(B)sT

    Input
    ------
        B : matrix
            Matrix B
        s : vector
            Vector s

    Output
    ------
        D : matrix
            Matrix log(B).s
    """
    S = np.transpose(np.matrix(s))
    B = np.matrix(np.log(B))
    (k1, k2) = np.shape(B)
    (k3, k4) = np.shape(S)
    if k2 == k3:
        D = np.zeros((k1,k4))
        for i in range(k1):
            for j in range(k4):
                for k in range(k2):
                    if np.log(S[k,j]) > - np.inf :
                        D[i,j] += B[i,k] *  S[k, j]
        return np.transpose(np.matrix(D))
    else:
        try:
            Shape()
        except Shape:
            print("Problème de dimension : ", (k1, k2), " avec ", (k3,k4))

def get_all_min(L):
    """
    Get argmin set of a list

    Input
    -----
        L : list
            List with numbers

    Output
    ------
        A : list
            List with indexes of minimum
    """
    A = []
    h = False
    for i in range(len(L)):
        if h == False:
            A.append(i)
            h = L[i]
        elif h - L[i] > 1e-10:
            h = L[i]
            A = [i]
        elif -1e-10 < h - L[i] < 1e-10:
            A.append(i)
    return A


def construct_A_report(s, k):
    """
    Construct A matrix for attentional focus (linked to report)

    Input
    -----
        s : vector
            Emotionnal hidden states
        k : integer
            Number of emotionnal states

    Output
    ------
        matrix
            A matrix for report attentional states
    """
    A = np.matrix(np.zeros((2, k)))
    for i in range(k):
        A[0, i] = s[i]
        A[1, i] = 1 - s[i]
    return A


def ambrep(es, rs, k):
    """
    Ambiguity for report states

    Input
    -----
        es: vector
            Emotionnal hidden states
        rs: vector
            Report hidden states
        k:  integer
            Number of emotions

    Output
    ------
        scalar
            Ambiguity for report states
    """
    s = 0
    for i in range(k):
        if not(np.log(es[i]) == -np.inf or np.log(1 - es[i]) == -np.inf):
            s += rs[i] * (es[i] * np.log(es[i]) + (1- es[i]) * np.log(1-es[i]))
    return s


def antinan(a):
    (m, n) = np.shape(np.matrix(a))
    S = np.matrix(a)
    for i in range(m):
        for j in range(n):
            if S[i,j] != S[i, j]:
                S[i,j] = 0
    return S
