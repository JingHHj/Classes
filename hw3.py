import numpy as np
import gym
import matplotlib.pyplot as plt; plt.ion()

class MDP(object):
  def __init__(self, P, nX, nU, gamma = 0.9):
    self.nX = nX # dimension of state space
    self.nU = nU # dimension of control space
    self.gamma = gamma 
    self.L = np.zeros((nX,nU))    # stage cost: SxA -> R   
    self.P = np.zeros((nX,nU,nX))
    for x in range(nX):
      for u in range(nU):
        nx = P[x][u][0]
        self.L[x,u] += P[x][u][1]
        self.P[x,u,nx] = 1

def value_iteration(mdp, num_iter):
    """
    V, pi = value_iteration(mdp, num_iter)
    """
    # initialize the policy and value
    pi = np.zeros((num_iter+1,mdp.nX),dtype='int') # (num,nX)
    V = np.zeros((num_iter+1,mdp.nX)) # (num,nX)

    # value iteration  
    for k in range(num_iter):
        Q = mdp.L + mdp.gamma * np.sum(mdp.P * V[k,None,None,:], axis=2) # nX x nA
          # nX*nA +  nX*nA(nX*nA*nX )
        pi[k+1,:] = np.argmin(Q, axis=1) # (,nX)
        V[k+1,:] = np.min(Q,axis=1) # (,nX)
        print(pi[k+1,:])
    return V, pi

def Qvalue_iteration(mdp, num_iter):
    """
    V, pi = value_iteration(mdp, num_iter)
    """
    # initialize the policy and value
    pi = np.zeros((num_iter+1,mdp.nX),dtype='int') # (num,nX)
    V = np.zeros((num_iter+1,mdp.nX)) # (num,nX)
    iall_sta = np.arange(mdp.nX)
    Qstar = np.zeros((num_iter+1,mdp.nX,mdp.nU)) # (num,nX,nU)   

    # value iteration  
    for k in range(num_iter):
        pi[k,:] = np.argmin(Qstar[k,:,:], axis=1) # (,nX)
        Vstar = np.min(Qstar[k,:,:],axis=1) # (,nX)
        # print(mdp.L[iall_sta,pi[k]].shape)
        # print(mdp.P[iall_sta,pi[k],:])
        # print(Vstar[:,None,None].shape)
        Q = mdp.L[iall_sta,pi[k]] + mdp.gamma * np.sum(mdp.P[iall_sta,pi[k],:] @ Vstar[:,None], axis=1) # (,nX)    
        V[k+1,:] = Q
        print(pi[k,:])
    return V, pi

def policy_iteration(mdp, num_iter):
    """
    Vpi, pi = policy_iteration(mdp, num_iter)
    """   
    # terminal and nonterminal states  
    ntrm_I = np.eye(mdp.nX)
    iall_sta = np.arange(mdp.nX)

    # initialize the policy and value
    pi = np.zeros((num_iter+1,mdp.nX),dtype='int')
    Vpi = np.zeros((num_iter+1,mdp.nX))
    # policy iteration 
    for k in range(num_iter):
    
        # Policy Evaluation
        Ppi = mdp.P[iall_sta, pi[k]]
        A = ntrm_I - mdp.gamma * Ppi
        b = mdp.L[iall_sta, pi[k]]
        Vpi[k,:] = np.linalg.solve(A, b)

        Qpi = mdp.L + mdp.gamma * np.sum(mdp.P * Vpi[k,None,None,:], axis=2)
        pi[k+1,:] = np.argmin(Qpi, axis=1) 
        print(pi[k+1,:])
    
    # Final Policy Evaluation
    A = ntrm_I - mdp.gamma 
    b = mdp.L[iall_sta, pi[k]]
    Vpi[k,:] = np.linalg.solve(A, b)
    return Vpi, pi


def displayValuesText(V,pi):
  print("Iteration | max|V-Vprev| | # chg actions | V[0]")
  print("----------+--------------+---------------+---------")
  for k in range(V.shape[0]-1):
    max_diff = np.abs(V[k+1] - V[k]).max()
    nChgActions=(pi[k+1] != pi[k]).sum()
    print("%4i      | %6.5f      | %4s          | %6.5f"%(k+1, max_diff, nChgActions, V[k+1,0]))
  print("----------+--------------+---------------+---------\n")   


"""
    0: north
    1: west
    2: south
    3: east
    (result, cost)

    0  1  2  3  4
    5  6  7  8  9
    10 11 12 13 14
    15 16 17 18 19
    20 21 22 23 24

      0
    1 a 3
      2
"""


def init(nX,nU):
    """
        initialize state space
    """
    p = {}
    for state in range(nX):
        q = {}
        if state == 1:
            x = (21,-10)
            q = {0: x,1:x,2:x,3:x}
            p[state] = q
            continue
        elif state == 3:
            x = (13,-5)
            q = {0: x,1:x,2:x,3:x}
            p[state] = q
            continue
        else: 
            for action in range(nU):
                match action:
                    case 0: # north
                        if state <= 4:
                            q[action] = (state ,1)
                            continue
                        else:
                            q[action] = (state -5,0)
                    case 1: # west
                        if state%5 == 0:
                            q[action] = (state ,1)
                            continue
                        else: 
                            q[action] = (state -1,0)
                    case 2: # south
                        if state  >= 20:
                            q[action] = (state ,1)
                            continue
                        else:
                            q[action] = (state +5,0)
                    case 3: # east
                        if state %5 == 4:
                            q[action] = (state ,1)
                            continue
                        else:
                            q[action] = (state + 1,0)
        p[state] = q

    # for state in range(nX):
    #     print(state,"  ",p[state])

    return p


if __name__ == "__main__":
    nX = 5*5
    nU = 4
    P = init(nX,nU)
    num_iter = 100
    mdp = MDP(P,nX,nU)

    # V1,pi1 = value_iteration(mdp, num_iter)
    # displayValuesText(V1,pi1)

    # V2,pi2 = policy_iteration(mdp, num_iter)
    # displayValuesText(V2,pi2)

    # V3,pi3 = Qvalue_iteration(mdp, num_iter)
    # displayValuesText(V3,pi3)
    
    
