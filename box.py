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
    obs:        matrix
        Observations
    time:       integer
        Time of observations
    number:     integer
        Number of outcomes
    globs:      dict (integer -> integer list)
        List of policy to separate observations by attentional focus
    obsglob:    dict (integer -> matrix)
        Observations according a attentional focus
    obsreg:     matrix (integer * integer -> float)
        Observation regrouping for each focus
    labels:     string list
        Name of outcomes
    """

    def __init__(self, obs=np.matrix([]), globalo=None, obsr=np.matrix([])):
        """
        Initialization of observation

        Input
        -----
        obs:        matrix
            Observations
        globalo:    dict (integer -> integer list)
            List of policy to separate observations by attentional focus
        """
        super(Observation, self).__init__()
        self.obs = np.matrix(obs)
        (t, outcomes) = np.shape(obs)
        self.time = t
        self.number = outcomes
        if globalo == None:
            self.globs = None
            self.obsglob = {0 : self.obs}
            self.obsreg = self.obs
        else:
            self.obsreg = obsr
            self.globs = globalo
            self.obsglob = {}
            for k in globalo:
                l = len(globalo[k])
                o = np.matrix(pm.epsilon * np.ones((t, l)))
                for h in range(l):
                    o[:, h] += self.obs[:, globalo[k][h]]
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
                o = np.matrix(pm.epsilon * np.ones((t, l)))
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

    def set_labels(self, labels=[]):
        """
        Set names of outcomes.
        (For displaying)

        Input
        -----
        labels: string list
            Name of outcomes
        """
        self.labels = [''] + labels

    def display_graph(self, title="Observations"):
        """
        Display outcomes via matplotlib interface

        Input
        -----
        title: string
            Name of the figure
        """
        dp.plot_states(self.obs, title, self.labels)


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
    A:              matrix dict
        A matrix (Observation/State) for each action
    Af:             matrix
        A matrix (O/S) for focus (non report) states
    B:              matrix dict
        B matrix (States transitions) for each action
    B_emo:          matrix
        B matrix (States transitions) for emotions
    D:              vector
        Initial states probabilities
    fstates:        matrix (integer * integer -> float)
        Final states after choosing policies
    labels:         string list
        Name of outcomes
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
        self.fstates = np.matrix([])
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
        self.states = np.ones((self.N_policy, t, self.N_states))
        self.states[:,:,0:self.N_states_emo] = self.states[:,:,0:self.N_states_emo]/self.N_states_emo
        self.states[:,:,self.N_states_emo:] = self.states[:,:,self.N_states_emo:]/self.N_states_act
        self.fstates = pm.epsilon * np.ones((t, self.N_states))
        for i in range(self.N_policy):
            self.states[i, 0, :] = D[:]


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
            S = np.matrix(pm.epsilon * np.ones(self.N_policy, obs.time, self.N_states))
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
        if type(Af) != type(None):
            self.Af = Af
        if type(A) != type(None):
            for k in A:
                if k in self.policy.keys():
                    self.A[k] = A[k]
                else:
                    print("Warning: Be careful with ", k, \
                        " index in A dict, it's not in policies list")
        if type(B) != type(None):
            for k in B:
                if k in self.policy.keys():
                    self.B[k] = B[k]
                else:
                    print("Warning: Be careful with ", k, \
                        " index in B dict, it's not in policies list")
        if type(B_emo) != type(None):
            self.B_emo = B_emo
        self.D = D

    def belief_updating(self):
        i = 0
        for p in self.policy:
            (G, H) = gd.gradF_v(self.states[i, :, 0:self.N_states_emo], \
                        self.states[i, :, self.N_states_emo:], \
                        self.obs.obsglob[p], self.obs.obsreg, \
                        self.A[p], self.Af, self.D, self.B_emo, self.B[p], \
                        self.time, self.obs.time, self.N_states_emo)
            G = mt.antinan(G)
            H = mt.antinan(H)
            k = 1
            while (mt.norm(G) + mt.norm(H)) > 0.0001:
                k += 1
                self.states[i, :, 0:self.N_states_emo] = self.states[i, :, 0:self.N_states_emo] - 1/k * G
                self.states[i, :, self.N_states_emo:] = self.states[i, :, self.N_states_emo:] - 1/k * H
                for h in range(self.time):
                    self.states[i,h, 0:self.N_states_emo] = mt.softmax(k*10* self.states[i, h, 0:self.N_states_emo])
                    self.states[i, h, self.N_states_emo:] = mt.softmax(k*20 * self.states[i, h, self.N_states_emo:])
                (G, H) = gd.gradF_v(self.states[i, :, 0:self.N_states_emo], \
                            self.states[i, :, self.N_states_emo:], \
                            self.obs.obsglob[p], self.obs.obsreg, \
                            self.A[p], self.Af, self.D, self.B_emo, self.B[p], \
                            self.time, self.obs.time, self.N_states_emo)
                G = mt.antinan(G)
                H = mt.antinan(H)
            i += 1

    def choose_policy(self):
        """
        Method to create the sequence of policies
        """
        act = fe.get_policies(self.states, self.A, self.obs.obsglob, \
                            self.policy, self.time, self.N_states_emo)
        self.fstates = fe.fix_s(self.states, act, \
                                self.time, self.N_states)


    def set_labels(self, labels=[]):
        """
        Set names of states. (For displaying)

        Input
        -----
        labels: string list
            Name of states
        """
        self.labels = [''] + labels

    def display_graph(self, title="Observations & States", alld=True):
        """
        Display states via matplotlib interface

        Input
        -----
        title:  string
            Name of the figure
        alld:   boolean
            Display states graph or states&observations graph
        """
        if alld:
            dp.plot_all(self.fstates, self.obs.obs, title, \
                self.labels, self.obs.labels)
        else:
            dp.plot_states(self.states, title, self.labels)
