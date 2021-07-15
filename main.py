"""
Fichier prinicpal
"""

import numpy as np
from scipy.special import softmax
import mathtools as mt
import parameters as pm
import matrices as mat
import display as dp

(D, A, B, C) = mat.import_matrices()
o = mat.observ_gen()
dp.plot_states(o, "Observations générées")
