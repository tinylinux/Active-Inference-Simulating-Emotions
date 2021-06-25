"""
Fichier principal
"""

import numpy as np
from scipy.special import softmax
import matools as mt
import free_energy as fe
import parameters as pm

A = []
B_s = [np.loadtxt("matrices/B_1.m")]
B_e = np.loadtxt("matrices/B_emo.m")
for i in range(2,11):
    A.append(np.loadtxt("matrices/A_" + str(i) + ".m"))
    B.append(np.loadtxt("matrices/B_" + str(i) + ".m"))
D = np.loadtxt("matrices/D.m")
C = np.loadtxt("matrices/C.m")
