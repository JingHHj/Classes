import numpy as np
R1 = np.array([
    [0.173,-0.147,0.974],
    [0.974,0.173,-0.147],
    [-0.147,0.974,0.173]
])
R2 = np.array([
    [0.707,-0.707,0],
    [0.707,0.707,0],
    [0,0,1]
])
R3 = np.array([
    [0.707,-0.653,0.271],
    [0.707,0.653,-0.271],
    [0,0.383,0.924]
])
Q = 2.*(R1+R2+R3)

U,sigma,V = np.linalg.svd(Q)
sigma = np.diag(sigma)

det = np.linalg.det(U@(V.T))
R_star = U@np.array([
    [1,0,0],
    [0,1,0],
    [0,0,det]
                    ])@(V.T)

print("optimal solution for n = 3 is \n",R_star)