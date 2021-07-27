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
import free_energy as fe

(D, A, Af, B, B_emo, C) = mat.import_matrices()
(O, Og) = mat.observ_gen()
#dp.plot_states(O, "Observations générées")

s = np.ones((pm.N_policy, pm.T, pm.N_states)) / pm.N_states_emo

i = 0
for p in pm.Policy:
    o = np.zeros((pm.T, len(pm.Policy[p])))
    for k in range(len(pm.Policy[p])):
        o[:,k] = O[:,pm.Policy[p][k]]
    (G, H) = gd.gradF_v(s[i, :, 0:pm.N_states_emo], s[i, :, pm.N_states_emo:], o, Og, A[p], Af, D, B_emo, B[p], pm.T)
    k = 1
    while (mt.norm(G) + mt.norm(H)) > 0.00001:
        k += 1
        s[i, :, 0:pm.N_states_emo] = s[i, :, 0:pm.N_states_emo] - 1/k * G
        s[i, :, pm.N_states_emo:] = s[i, :, pm.N_states_emo:] - 1/k * H
        for h in range(pm.T):
            s[i, h, 0:pm.N_states_emo] = mt.softmax(k *10* s[i, h, 0:pm.N_states_emo])
            s[i, h, pm.N_states_emo:] = mt.softmax(k*10* s[i, h, pm.N_states_emo:])
        (G, H) = gd.gradF_v(s[i, :, 0:pm.N_states_emo], s[i, :, pm.N_states_emo:], o, Og, A[p], Af, D, B_emo, B[p], pm.T)
    i += 1

fs = np.ones((pm.T, pm.N_states)) # Final states

#for t in range(pm.T):
#    for k in pm.Policy:
#        print(fe.G(s[0, t, 0:pm.N_states_emo], A[k], C[k]))

for i in range(pm.N_policy):
    dp.plot_states(np.transpose(s[i, :, :]), "Policy " + str(i))

Actions = fe.get_policies(s, A, C)
fs = fe.fix_s(s, Actions)
#dp.plot_states(np.transpose(fs), "Hidden states")
dp.plot_all(fs, O)
