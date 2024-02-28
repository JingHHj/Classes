import numpy as np
import modern_robotics as mr

def get_skew_symmetric(v):
    skew_symmetric_matrix = np.array([
        [0.,-v[2],v[1]],
        [v[2],0.,-v[0]],
        [-v[1],v[0],0.]
    ])
    return skew_symmetric_matrix

def get_inv(matrix):
    matrix_inv = np.vstack((
        np.hstack((
            matrix[:3,:3].T, -matrix[:3,:3].T@(matrix[:3,3].reshape(3,1)) 
            )),
        np.array([0.,0.,0.,1.])
    ))
    return matrix_inv

def get_adjoint_matrix(matrix):
    adjoint_matrix = np.vstack((
            np.hstack((matrix[:3,:3],np.zeros((3,3)))),
            np.hstack((np.dot(get_skew_symmetric(matrix[:3,3]),matrix[:3,:3]),
                                          matrix[:3,:3]))
    ))
    return adjoint_matrix 

def space2body(screw_s,ee_pose):
    ee_pose_inv = get_inv(ee_pose)
    return get_adjoint_matrix(ee_pose_inv)@screw_s

 


W1 = 0.109
W2 = 0.082 
L1 = 0.425 
L2 = 0.392
H1 = 0.089 
H2 = 0.095 
M = np.array([
    [-1.,0.,0.,L1+L2],
    [0.,0.,1.,W1+W2],
    [0.,1.,0.,H1-H2],
    [0.,0.,0.,1.]
])



# desired end-effector configuration
Tsd = np.array([
    [0.,1.,0.,-0.5],
    [0.,0.,-1.,0.1],
    [-1.,0.,0.,0.1],
    [0.,0.,0.,1.]
])
eomg = 0.001
ev = 0.001
thetalist0 = np.array([2.6,-1.,1.7,-0.691,-0.576,4.7])
# Bi = [AdM-1]Si
Si = np.array([
    [0., 0.,  0.,   0.,   0.,   0.],
    [0., 1.,  1.,   1.,   0.,   1.],
    [1., 0.,  0.,   0.,  -1.,   0.],
    [0.,-H1,-H1,   -H1,  -W1, H2-H1],
    [0., 0.,  0.,   0., L1+L2,  0.],
    [0., 0.,  L1, L1+L2,  0., L1+L2],
])

Bi = space2body(Si,M)





def IKinBodyIterations():
    steps = 3
    for i in range(steps):
        thetalist1,x = mr.IKinBody(Bi,M,Tsd,thetalist0,eomg,ev)
        thetalist0 = np.asarray(thetalist1)
        print("The iteration number: ",i)
        print("The joint vector: ",thetalist0)
        print("The current end-effector configuration Tsb(θi)",)
        print("The error twist Vb")
        print("the angular and linear error magnitudes ∥ωb∥ and ∥vb∥")
    return

print(thetalist0,x)


