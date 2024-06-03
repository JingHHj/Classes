import numpy as np
import cvxpy as cp

p = np.array([
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0.5,0.5],
    [0,1,0,0]
])
l = np.array([-10,1,2,3])
gamma = 0.9

n = p.shape[0]
w = np.array([1/4,1/4,1/4,1/4])
v0 = np.array([0,0,0,0])

A = p
b = l + gamma * p @ v0

c = w
A = np.eye(n)
# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c.T@x),
                 [A @ x <= b])
prob.solve()
print(x.value)

