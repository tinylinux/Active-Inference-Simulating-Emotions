"""
Fichier où sont definis les paramètres
"""

T = 5

Policy = {
    1:  [1,2],      # Attend to Valence
    #2:  [3,4],      # Attend to Arousal
    3:  [5,6],      # Attend to Motivation
    #4:  [7,8],      # Attend to Beliefs
    #5:  [9,10],     # Attend to Context
    6:  [11, 12],   # Report Sad
    #7:  [11, 12],   # Report Afraid
    #8:  [11, 12],   # Report Angry
    9:  [11, 12]    # Report Happy
}

Emotions = {
    1:  6,  # Sad
    #2:  7,  # Afraid
    #3:  8,  # Angry
    4:  9   # Happy
}

N_policy = 9
N_states_emo = 4
N_states_act = 10
N_states = N_states_emo + N_states_act

N_outcomes = 13
