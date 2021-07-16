"""
Fichier où sont definis les paramètres
"""

T = 5

Policy = {
    2:  [1,2],      # Attend to Valence
    #3:  [3,4],      # Attend to Arousal
    4:  [3,4],      # Attend to Motivation
    #5:  [7,8],      # Attend to Beliefs
    #6:  [9,10],     # Attend to Context
    7:  [5, 6],   # Report Sad
    #8:  [11, 12],   # Report Afraid
    #9:  [11, 12],   # Report Angry
    10:  [5, 6]    # Report Happy
}

Emotions = {
    1:  7,  # Sad
    #2:  8,  # Afraid
    #3:  9,  # Angry
    4:  10   # Happy
}

N_policy = 4
N_states_emo = 2
N_states_act = 5
N_states = N_states_emo + N_states_act

N_outcomes = 7
