"""
Fichier mettant en avant une classe de simulation d'émotions
"""

import numpy as np
from scipy.special import softmax
import mathtools as mt
import parameters as pm
import matrices as mat
import display as dp
import gradient as gd
import free_energy as fe

class Observation(object):
    """
    Classe concernant les observations
    """

    def __init__(self, obs=np.matrix([]), global=[]):
        super(Observation, self).__init__()
        self.obs = obs
        (t, outcomes) = np.shape(obs)
        self.time = t
        self.number = outcomes
        if global == []:
            self.globs = self.obs
        else:
            l = len(global)
            self.globs = np.matrix(np.zeros((t, l)))
            for k in range(len(global)):
                for i in global[k]:
                    self.globs[:, i] += self.obs[:, i]

    def upload(self, file):
        self.obs = np.matrix(np.loadtxt(file))
        (t, outcomes) = np.shape(self.obs)
        self.time = t
        self.number = outcomes

    def add(self, table):
        T = np.matrix(table)
        (t, outcomes) = np.shape(T)
        S = np.matrix(np.zeros(self.time + T, outcomes))
        S[0:self.time, :] = self.obs[:, :]
        S[self.time:, :] = T[:, :]
        self.obs = S
        self.time = self.time + T


class ActInf(object):
    """
    Classe d'inférence active pour la simulation d'émotions
    pour un individu
    """

    def __init__(self, arg):
        super(ActInf, self).__init__()
        self.arg = arg

    def upd_obs(self, obs):
        self.obs = np.array([])
