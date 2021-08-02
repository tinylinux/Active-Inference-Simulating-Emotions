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

    Attributes
    ----------
    obs:    matrix
        Observations
    time:   integer
        Time of observations
    number: integer
        Number of outcomes
    globs: dict (integer -> integer list)
        List of policy to separate observations by attentional focus
    obsglob:  dict (integer -> matrix)
        Observations according a attentional focus
    """

    def __init__(self, obs=np.matrix([]), global={}):
        """
        Initialization of observation

        Input
        -----
        obs:    matrix
            Observations
        global: dict (integer -> integer list)
            List of policy to separate observations by attentional focus
        """
        super(Observation, self).__init__()
        self.obs = np.matrix(obs)
        (t, outcomes) = np.shape(obs)
        self.time = t
        self.number = outcomes
        if global == {}:
            self.globs = {}
            self.obsglob = {0 : self.obs}
        else:
            self.globs = global
            self.obsglob = {}
            for k in global:
                l = len(global[k])
                o = np.matrix(pm.epsilon * np.ones((t, l)))
                for h in range(l):
                    o[:, h] += self.obs[:, global[k][h]]
                self.obsglob[k] = np.matrix(o)

    def upload(self, file):
        """
        Upload observations from a file

        Input
        -----
        file:   string (path)
            Path of a observation file
        """
        self.obs = np.matrix(np.loadtxt(file))
        (t, outcomes) = np.shape(self.obs)
        self.time = t
        self.number = outcomes
        if self.globs == {}:
            self.obsglob = {0 : self.obs}
        else:
            self.obsglob = {}
            for k in self.globs:
                l = len(self.globs[k])
                o = np.matrix(np.zeros((t, l)))
                for h in range(l):
                    o[:, h] += self.obs[:, self.globs[k][h]]
                self.obsglob[k] = np.matrix(o)

    def add(self, table):
        """
        Add an observation for a period

        Input
        -----
        table:  matrix
            Observations done after saved observations
        """
        T = np.matrix(table)
        (t, outcomes) = np.shape(T)
        S = np.matrix(np.zeros(self.time + T, outcomes))
        S[0:self.time, :] = self.obs[:, :]
        S[self.time:, :] = T[:, :]
        self.obs = S
        self.time = self.time + T
        if self.globs == {}:
            self.obsglob = {0 : self.obs}
        else:
            self.obsglob = {}
            for k in self.globs:
                l = len(self.globs[k])
                o = np.matrix(np.zeros((t, l)))
                for h in range(l):
                    o[:, h] += self.obs[:, self.globs[k][h]]
                self.obsglob[k] = np.matrix(o)


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
    A:      matrix dict
        A matrix (Observation/State) for each action
    Af:     matrix
        A matrix (O/S) for focus (non report) states
    B:      matrix dict
        B matrix (States transitions) for each action
    B_emo:  matrix
        B matrix (States transitions) for emotions
    D:      vector
        Initial states probabilities
    """

    def __init__(self):
        """
        Initialization.
        """
        super(ActInf, self).__init__()
        self.time = 0
        self.N_states_emo = 0
        self.N_states_act = 0
        self.N_states = 0
        self.N_policy = 0
        self.states = np.matrix([])
        self.policy = {}
        self.A = {}
        self.Af = np.matrix([])
        self.B = {}
        self.B_emo = np.matrix([])
        self.obs = Observation()

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

        Warning
        -------
            It doesn't duplicate the observations.
            Be careful when you change the variable of the observation entity.

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

    def set_matrices(self, A=None, Af=None, B=None, B_emo=None, D=pm.D):
        """
        Set matrices for transition and observations.

        Warning
        -------
            It doesn't duplicate the matrix.
            Be careful when you change the variable of the matrix root
            If you intend to change your matrix variable, create a duplicate
            of this (with `np.matrix` command) like :
                `I.set_matrices(np.matrix(A), ...)`

        Input
        -----
        A:      matrix dict
            A matrix (Observation/State) for each action
        Af:     matrix
            A matrix (O/S) for focus (non report) states
        B:      matrix dict
            B matrix (States transitions) for each action
        B_emo:  matrix
            B matrix (States transitions) for emotions
        D:      vector
            Initial states probabilities
        """
        if Af != None:
            self.Af = Af
        if A != None;
            for k in A:
                if k in self.policy.keys():
                    self.A[k] = A[k]
                else:
                    print("Warning: Be careful with ", k, \
                        " index in A dict, it's not in policies list")
        if B != None:
            for k in B:
                if k in self.policy.keys():
                    self.B[k] = B[k]
                else:
                    print("Warning: Be careful with ", k, \
                        " index in B dict, it's not in policies list")
        if B_emo != None:
            self.B_emo = B_emo
        self.D = D
