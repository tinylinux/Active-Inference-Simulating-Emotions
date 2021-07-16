"""
Fichier prinicpal
"""

import numpy as np
from scipy.special import softmax
import mathtools as mt
import parameters as pm
import matrices as mat
import display as dp
import gradient as gd

(D, A, B, B_emo, C) = mat.import_matrices()
O = mat.observ_gen()
dp.plot_states(O, "Observations générées")

s = np.ones((pm.N_policy, pm.T, pm.N_states)) / pm.N_states_emo

i = 0
for p in pm.Policy:
    o = np.zeros((pm.T, len(pm.Policy[p])))
    for k in range(len(pm.Policy[p])):
        o[:,k] = O[:,pm.Policy[p][k]]
    G = gd.gradF_v(s[i, :, 0:pm.N_states_emo], o, A[p], D[0:2], B_emo, pm.T)
    print(G)
    k = 1
    while mt.norm(G) > 0.01:
        k += 1
        print(s[i,:,0:pm.N_states_emo], G)
        s[i, :, 0:pm.N_states_emo] = s[i, :, 0:pm.N_states_emo] - 1/k * G
        for h in range(pm.T):
            s[i, h, 0:pm.N_states_emo] = mt.softmax(k *10* s[i, h, 0:pm.N_states_emo])
        G = gd.gradF_v(s[i, :, 0:pm.N_states_emo], o, A[p], D[0:2], B_emo, pm.T)
        print(mt.norm(G))
    i += 1
print(s)
