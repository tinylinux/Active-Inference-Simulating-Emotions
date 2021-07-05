"""
Fichier pour afficher les résultats d'états
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_states(s, title):
    name = "gray"
    plt.matshow(s, cmap=plt.get_cmap(name))
    plt.title(title)
    plt.colorbar()
    plt.show()
