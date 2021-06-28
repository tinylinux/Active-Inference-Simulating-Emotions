"""
Fichier pour tester les algos
"""

import numpy as np
from scipy.special import softmax
import matools as mt

A = np.loadtxt("matrices/A_3.m")
B_e = np.loadtxt("matrices/B_emo.m")
B_s = np.loadtxt("matrices/B_3.m")
D = np.loadtxt("matrices/D.m")
C = np.loadtxt("matrices/C.m")

import gradient as grd

s0_emo = softmax(D[0:4])
s0_pos = softmax(D[4:])

s = [(np.array(s0_emo), np.array(s0_pos)) for i in range(5)]

o = [np.array(C) for i in range(3)]

Gd = np.array(grd.gradF_v(s, o, (B_e, B_s), A, (s0_emo, s0_pos)))
# print("Gd : ", Gd)
k = 1
while mt.norm(Gd.reshape((10))) > 0.001:
    for i in range(5):
        s[i] = (s[i][0] - 1/(2**k)*Gd[i][0],
                s[i][1] - 1/(2**k)*Gd[i][1])
        s[i] = (mt.softmax(20*s[i][0]), mt.softmax(20*s[i][1]))
    Gd = np.array(grd.gradF_v(s, o, (B_e, B_s), A, (s0_emo, s0_pos)))
    k += 1
    print(mt.norm(Gd.reshape((10))))
    # print(s)
