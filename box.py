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

    Attributes
    ----------
    time :          integer
        Duration of the active inference
    emotions :      dict (integer -> integer)
        Link emotion and report state
    N_states_emo :  integer
        Number of emotionnal states
    N_states_act :  integer
        Number of attentional focus states
    states :        matrix (integer * integer * integer -> float)
        Distribution over states
    policy :        dict (integer -> integer list)
        Policy with list of observations associed to each policy
    N_policy :      integer
        Number of policies
    obs :           observations
        Observations for this entity
    """

    def __init__(self):
        """
        Initialization.
        """
        super(ActInf, self).__init__()
        self.time = 0
        self.N_states_emo = 1
        self.N_states_act = 1
        self.N_states = 2
        self.N_policy = 1
        self.states = np.matrix([])

    def set_policy(self, policy):
        """
        Set policies

        Input
        -----
        policy :    dict (integer -> integer list)
            Policy with observations index attached to this policy
        """
        self.policy = policy
        self.N_policy = len(policy.keys())

    def set_states(self, emotions, nacts, t, D):
        """
        Set states

        Input
        -----
        emotions :  dict (integer -> integer)
            Link emotion and report state
        nacts :     integer
            Number of attentional focus state
        t :         integer
            Duration of active inference process
        D :         array
            Array of initial states
        """
        self.emotions = emotions
        self.N_states_emo = len(emotions.keys())
        self.N_states_act = nacts
        self.N_states = self.N_states_emo + self.N_states_act
        self.time = t
        self.states = np.matrix(np.zeros((self.N_policy, t, self.N_states)))
        for i in range(self.N_policy):
            for j in range(t):
                self.states[i, j, :] = D[:]

    def upd_obs(self, obs):
        """
        Update observations

        Input
        -----
        obs :     observation
            All observations done for a entity by a entity (itself or subject)
        """
        self.obs = obs
        if obs.time > self.time:
            S = np.matrix(np.zeros(self.N_policy, obs.time, self.N_states))
            S[:, 0:self.time, :] = self.states[:, :, :]
            self.time = T
