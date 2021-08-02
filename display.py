"""
Fichier pour afficher les résultats d'états
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import parameters as pm

def plot_states(s, title, labels=pm.States):
    name = "gray"

    fig, (ax1) = plt.subplots(1,1)
    fig.suptitle(title, fontsize=20)

    im1 = ax1.imshow(np.transpose(s), cmap=plt.get_cmap(name), aspect='auto')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    ax1.xaxis.set_visible(False)
    ax1.set_yticklabels(labels)

    plt.tight_layout()
    # Make space for title
    plt.subplots_adjust(top=0.85)
    plt.show()


def plot_all(s, o, title, labels=pm.States, labelo=pm.Outcomes):
    name = "gray"

    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle(title, fontsize=20)

    ax1.set_title('Observations')
    im1 = ax1.imshow(np.transpose(o), cmap=plt.get_cmap(name), aspect='auto')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    ax1.xaxis.set_visible(False)
    ax1.set_yticklabels(labelo)

    ax2.set_title('States')
    im2 = ax2.imshow(np.transpose(s), cmap=plt.get_cmap(name), aspect='auto')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2)
    ax2.xaxis.set_visible(False)
    ax2.set_yticklabels(labels)

    plt.tight_layout()
    # Make space for title
    plt.subplots_adjust(top=0.85)
    plt.show()
