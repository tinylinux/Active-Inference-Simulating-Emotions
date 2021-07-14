"""
Fichier contenant la fonction de calcul du gradient de l'énergie libre
"""

import numpy as np
import matools as mt

def gradF_s(s_emo, o, B_r, B_e, A, D):
    """
    Calcul du gradient de F (énergie libre) selon s

    Input
    ------
        s_emo : [list : vector]
            Etats internes - Emotions
        o : [list : vector]
            Observations/ Etats physiologiques
        A : matrix
            Lien Etats internes et physiologiques
        D : vector * vector
            Etats initiaux

    Output
    -------
        [list : vector * vector]
    """
    logA = np.matrix(np.log(A))
    #logA = np.matrix(mt.logA(A, s_emo, o))
    logB_r = np.matrix(np.log(B_r))
    logB_e = np.matrix(np.log(B_e))

    #Emo = []
    #Res = []
    n = len(s_emo)
    t = len(o)
    ke = np.shape(s_emo[0])

    grad = np.zeros((n, ke[0]))

    if n > 1:
        v_emo = np.ones(ke) + np.log(s_emo[0]) + mt.xlogy(np.matrix(o[0]), A) + np.log(D[0])
        print(np.matrix(o[0]))
        print(A)
        print(mt.xlogy(np.matrix(o[0]), A))
        print("------")
        grad[0,:] = np.array(v_emo)
    for i in range(1, min(n-1, t)):
        v_emo = np.ones(ke) + np.log(s_emo[i]) + mt.xlogy(np.matrix(o[i]), A)
        grad[i,:] = np.array(v_emo)
    for i in range(t, n-1):
        v_emo = np.ones(ke) + np.log(s_emo[i])
        grad[i,:] = np.array(v_emo)
    if n > 1:
        if t >= n:
            v_emo = np.matrix(np.ones(ke) + np.log(s_emo[n-1])) + mt.xlogy(np.matrix(o[n-1]), A)
        else:
            v_emo = np.matrix(np.ones(ke) + np.log(s_emo[n-1]))
        grad[n-1,:] = np.array(v_emo)
    return grad


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
        s_emo.append(k[0:4])
        s_res.append(k[4:])

    f = gradF_s(s_emo, o, B[1], B[0], A, D)

    gse = f[:,0:4]
    gsr = f[:,4:]

    L = []
    for i in range(len(s)):
        d_emo = np.diag(s_emo[i])
        #d_res = np.diag(s_res[i])
        c_emo = mt.inner(s_emo[i], s_emo[i])
        #c_res = mt.inner(s_res[i], s_res[i])
        v_emo = np.dot((d_emo - c_emo), np.transpose(gse[i]))
        #v_res = np.dot((d_res - c_res), np.transpose(gsr[i]))
        L.append(np.array(v_emo).reshape(4))

    return L
