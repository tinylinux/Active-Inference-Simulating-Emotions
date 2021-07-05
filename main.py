"""
Fichier principal
"""

import numpy as np
from scipy.special import softmax
import matools as mt
import free_energy as fe
import parameters as pm
import outcomes as ot
import display as dp

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
s_choosen = np.zeros((pm.T, pm.N_states))

## Initialisation des états
for j in range(pm.N_policy):
    for i in range(pm.T):
        s[j,i] = D[:]
        s[j,i,pm.N_states_emo] = 0
        s[j,i,pm.N_states_emo + j] = 1
        s[j,i,:] = softmax(s[j,i,:])

o = ot.get_outcomes()
dp.plot_states(np.transpose(o), "Observations")

## Descente de gradient pour chaque politique
for j in range(pm.N_policy):
    Gd = np.array(grd.gradF_v(s[j], o, (B_e, B_s[j+1]), A[j], (s0_emo, s0_pos)))
    k = 1
    while mt.norm(Gd.reshape((10))) > 0.001:
        for i in range(pm.T):
            s[j,i,0:4] = s[j,i,0:4] - 1/(2**k)*Gd[i][0]
            s[j,i,4:]  = s[j,i,4:]  - 1/(2**k)*Gd[i][1]
            s[j,i,0:4] = mt.softmax(20*k*s[j,i,0:4])
            s[j,i,4:]  = mt.softmax(20*k*s[j,i,4:])
        Gd = np.array(
            grd.gradF_v(s[j], o, (B_e, B_s[j+1]), A[j], (s0_emo, s0_pos))
            )
        k += 1

## Choix de l'action
for j in range(pm.T):
    p = fe.choose(A, s[:,j,0:4], softmax(C))
    print(p)
    s_choosen[j,:] = s[p,j,:]

print(s_choosen)
dp.plot_states(np.transpose(s_choosen), "Internal states")
