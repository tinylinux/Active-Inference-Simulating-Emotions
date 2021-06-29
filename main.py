"""
Fichier principal
"""

import numpy as np
from scipy.special import softmax
import matools as mt
import free_energy as fe
import parameters as pm
import outcomes as ot

A = []
B_s = [np.loadtxt("matrices/B_1.m")]
B_e = np.loadtxt("matrices/B_emo.m")
for i in range(2,11):
    A.append(np.loadtxt("matrices/A_" + str(i) + ".m"))
    B_s.append(np.loadtxt("matrices/B_" + str(i) + ".m"))
D = np.loadtxt("matrices/D.m")
C = np.loadtxt("matrices/C.m")

import gradient as grd

s0_emo = softmax(D[0:4])
s0_pos = softmax(D[4:])

s = np.zeros((pm.N_policy,pm.T,pm.N_states))

## Initialisation des états
for j in range(pm.N_policy):
    for i in range(pm.T):
        s[j,i] = softmax(D)

o = ot.get_outcomes()

## Descente de gradient pour chaque politique
for j in range(pm.N_policy):
    print("Politique n°", j, " : ")
    Gd = np.array(grd.gradF_v(s[j], o, (B_e, B_s[j+1]), A[j], (s0_emo, s0_pos)))
    k = 1
    print(mt.norm(Gd.reshape((pm.T * 2))))
    while mt.norm(Gd.reshape((10))) > 0.001:
        for i in range(pm.T):
            s[j,i,0:4] = s[j,i,0:4] - 1/(2**k)*Gd[i][0]
            s[j,i,4:]  = s[j,i,4:]  - 1/(2**k)*Gd[i][1]
            s[j,i,0:4] = mt.softmax(50*s[j,i,0:4])
            s[j,i,4:]  = mt.softmax(50*s[j,i,4:])
        Gd = np.array(
            grd.gradF_v(s[j], o, (B_e, B_s[j+1]), A[j], (s0_emo, s0_pos))
            )
        k += 1
        print(mt.norm(Gd.reshape((pm.T * 2))))

## Choix de l'action
