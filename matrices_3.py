"""
Generate approximative matrices
"""

import numpy as np
from scipy.special import softmax

B = np.loadtxt("matrices_brut/B_1.m")
b = np.ones(np.shape(B))
for j in range(10):
    b[:,j] = softmax(10 * B[:,j])
np.savetxt("matrices/B_1.m", b)

for i in range(2,11):
    A = np.loadtxt("matrices_brut/A_" + str(i) + ".m")
    B = np.loadtxt("matrices_brut/B_" + str(i) + ".m")
    a = np.ones(np.shape(A))
    b = np.ones(np.shape(B))
    for j in range(4):
        a[:,j] = softmax(100 * A[:,j])
    for j in range(10):
        b[:,j] = softmax(10 * B[:,j])
    np.savetxt("matrices/A_" + str(i) + ".m", a)
    np.savetxt("matrices/B_" + str(i) + ".m", b)


B_emo = np.loadtxt("matrices_brut/B_emo.m")
b_emo = np.ones(np.shape(B_emo))
for j in range(4):
    b_emo[:,j] = softmax(10 * B_emo[:,j])
np.savetxt("matrices/B_emo.m", b_emo)
