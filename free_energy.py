"""
Fichier traitant de la partie minimisation de l'énergie libre
Pour le calcul de G, on s'appuie sur l'Appendix D de l'article - A synthesis
"""

import numpy as np
import matools as mt
import display as dp
import time

def G(A, s, C, i):
    """
    Fonction permettant de calculer l'énergie libre espérée

    Input
    -----
        A : matrix
            Matrice A
        s : vector
            Vecteur des états internes
        C : vector
            Vecteur des préférences sur les outcomes

    Output
    ------
        scalar
            L'énergie libre espérée
    """
    H = -1 * np.diag(mt.xlogy(np.transpose(np.matrix(A)), np.matrix(A)))
    Ainv = np.matrix(np.array(A, dtype='f') ** -1)
    A0inv = np.matrix(np.array(mt.a0(A), dtype='f') ** -1)
    W = 1/2 * mt.subinf(Ainv, A0inv)
    As = np.dot(A, s)

    R1 = np.sum(mt.xlogx(As))
    R2 = np.sum(mt.xlogy(np.matrix(As), np.transpose(np.matrix(C))))
    # R = np.inner(As, mt.subinf(np.log(As), np.log(C)))
    R = R1 - R2
    S2 = np.inner(H, s)
    Ws = np.dot(W, s)
    S3 = np.inner(As, Ws)

    return R + S2


def pi(A, s, C, pi):
    """
    Calcul du vecteur de politique pour le choix de l'action

    Input
    -----
        A : [list : matrix]
            Matrice A
        s : array, dim=2
            Vecteur des états internes
        C : vector
            Vecteur des préférences sur les outcomes
        pi : int
            Nombre de politiques

    Output
    ------
        scalar
            L'énergie libre espérée
    """
    S = np.array([-1 * G(A[i], s[i], C, i) for i in range(pi)])
    return (np.argmin(S),mt.softmax(S))


def choose(A, s, C):
    """
    Choix de la meilleure politique

    Input
    -----
        A : [list : matrix]
            Matrice A
        s : array, dim=2
            Vecteur des états internes

    Output
    ------
        int
            numéro de la politique
    """
    (p, q) = pi(A, s, C, 9)
    #p = np.argmin(q)
    return (p,q)
