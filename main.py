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

o = ot.get_outcomes()
