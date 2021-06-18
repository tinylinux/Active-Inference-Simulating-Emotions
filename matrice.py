"""
Fichier d'import des matrices pour le modèle génératif
"""

import numpy as np
import matools as mt

# Politiques
P_START = 0
P_VALENCE = 1
P_AROUSAL = 2
P_MOTIVATION = 3
P_BELIEFS = 4
P_CONTEXT = 5
P_R_SAD = 6
P_R_AFRAID = 7
P_R_ANGRY = 8
P_R_HAPPY = 9
P_n = 10

# Observations
N_O = 13

# Matrice de départ
D = np.loadtxt("matrices/D.m")

# Matrices de transition
B = []
B_emo = np.loadtxt("matrices/B_emo.m")
for i in range(P_n):
    Bc = np.loadtxt("matrices/B_" + str(i+1) + ".m")
    B.append(Bc)

# Matrice etats/outcomes
A = []
for i in range(P_n):
    Ac = np.loadtxt("matrices/A_" + str(i+1) + ".m")
    A.append(Ac)

# Vecteur C
C = np.softmax(np.array([5 for i in range(11)] + [10, 0]))

# Vecteur 
