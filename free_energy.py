"""
Fichier traitant de la partie minimisation de l'énergie libre
Pour le calcul de G, on s'appuie sur l'Appendix D de l'article - A synthesis
"""

import numpy as np
import scipy as sp
import matools as mt

def G(A, s, C):
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
    H = -1 * np.diag(np.inner(A, np.log(A)))
    Ainv = np.matrix(np.array(A, dtype='f') ** -1)
    A0inv = np.matrix(np.array(mt.a0(A), dtype='f') ** -1)
    W = 1/2 * (Ainv - A0inv)
    R = np.inner(np.dot(A, s), (np.log(np.dot(A, s)) - np.log(C)))
    return R + np.inner(H, s) - np.inner(np.dot(A, s), np.dot(W, s))

def pi(A, s, C, pi):
    """
    Calcul du vecteur de politique pour le choix de l'action
    
    Input
    -----
        A : matrix
            Matrice A
        s : [list : vector]
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
    return mt.softmax(np.array([-1 * G(A[i], s[i], C) for i in range(1, pi+1)]))
