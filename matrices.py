"""
Fichier de génération des matrices pour le modèle génératif
"""

import numpy

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

# Matrice de départ
D = np.concatenate(1/4 * np.ones(4), np.ones(1), np.zeros(9))

# Matrice de transition entre états
B_emot = np.identity(4)

def B_ress(policy):
    """
    Matrice de transition suivant la politique
    """
    M = np.identity((9,9))
    for i in range(6):
        M[i,i] = 0
    for j in range(6):
        M[policy, j] = 1

# Matrice d'outcomes vers états
def A(policy):
    """
    Matrice de transition suivant la politique
    """
    M = np.zeros((13,4))
    if policy > 5:
        M[policy-6, -2] = 1
        for i in range(4):
            if not(i == policy - 6):
                M[i, -1] = 1
    elif policy == 0:
        for i in range(4):
            M[i, 0] = 1
    elif policy == 1:
    
