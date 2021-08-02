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
from box import Observation, ActInf

(D, A, Af, B, B_emo, C) = mat.import_matrices()
(O, Og) = mat.observ_gen()

Obs = Observation(O, pm.Policy, Og)
Obs.set_labels(pm.Outcomes[1:])
#Obs.display_graph()

Ai = ActInf()
Ai.set_policy(pm.Policy)
Ai.set_states(pm.Emotions, pm.N_states_act, pm.T, D)
Ai.upd_obs(Obs)
Ai.set_matrices(A, Af, B, B_emo, D)
Ai.belief_updating()
Ai.choose_policy()
Ai.set_labels(pm.States[1:])
Ai.display_graph()
