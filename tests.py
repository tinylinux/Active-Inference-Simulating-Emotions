"""
Fichier pour tester les algos
"""

import numpy as np
from scipy.special import softmax

A = np.loadtxt("matrices/A_3.m")
B_e = np.loadtxt("matrices/B_emo.m")
B_s = np.loadtxt("matrices/B_3.m")
D = np.loadtxt("matrices/D.m")
C = np.loadtxt("matrices/C.m")

import gradient as grd

s0_emo = D[0:4]
s0_pos = D[4:]

s = [(np.array(s0_emo), np.array(s0_pos)) for i in range(5)]

o = [np.array(C) for i in range(3)]

print("s : ", s)
Gd = np.array(grd.gradF_v(s, o, (B_e, B_s), A, (s0_emo, s0_pos)))
print("Gd : ", Gd)
k = 0
while np.linalg.norm(Gd) < 1:
    for i in range(5):
        s[i] = (s[i][0] + Gd[i][0], s[i][1] + Gd[i][1])
    Gd = np.array(grd.gradF_v(s, o, (B_e, B_s), A, (s0_emo, s0_pos)))
    k += 1
    print("k = ", k)
    print("s : ", s)
    print("Gd : ", Gd)
