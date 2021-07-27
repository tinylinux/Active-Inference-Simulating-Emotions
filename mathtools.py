"""
Fichier de module mathématiques
"""

import numpy as np
import scipy as sp

from numpy.linalg import norm
from scipy.special import softmax
from scipy.special import digamma
from scipy.special import gamma

log = np.log
gamma = gamma
psi = digamma
softmax = softmax

norm = norm

block_diag = sp.linalg.block_diag

class Shape(Exception):
    pass

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
        elif h > L[i]:
            h = L[i]
            A = [i]
        elif h == L[i]:
            A.append(i)
        print(L[i])
    return A

def construct_A_report(s, k):
    A = np.matrix(np.zeros((2, k)))
    print(s)
    for i in range(k):
        print(s[i])
        A[0, i] = s[i]
        A[1, i] = 1 - s[i]
    return A
