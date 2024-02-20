import numpy as np

def VecTose3(screw,theta):
    """
    In take the screw and the corresponding rotation angel theta
    Returns the corresponding 4x4 transformation matrix.
    """
    skew_symmetric = VecToso3(screw[:3])
    # getting the rotation matrix
    R = np.eye(3) + np.sin(theta)*skew_symmetric + (1. - np.cos(theta))*np.dot(skew_symmetric,skew_symmetric)
    # getting the translation vector
    G = np.eye(3)*theta + (1. - np.cos(theta))*skew_symmetric + (theta - np.sin(theta))*np.dot(skew_symmetric,skew_symmetric)
    P = G@(screw[3:].reshape(3,1))
    # put them togehter and output them
    temp = np.hstack((R,P))
    return np.vstack((temp,np.array([0.,0.,0.,1.]))) 

def VecToso3(omega):
    """
    Takes a 3-vector (angular velocity).
    Returns the skew symmetric matrix in so(3).
    """

    return np.array([[0,-omega[2],omega[1]],
                     [omega[2],0,-omega[0]],
                     [-omega[1],omega[0],0]
                    ])

def lab2(theta):
    # defining all the screw axes
    S1 = np.array([0.,0.,1.,-300.,0.,0.])
    S2 = np.array([0.,1.,0.,-240.,0.,0.])
    S3 = np.array([0.,1.,0.,-240.,0.,244.])
    S4 = np.array([0.,1.,0.,-240.,0.,457.])
    S5 = np.array([0.,0.,-1.,169.,457.,0.])
    S6 = np.array([0.,1.,0.,-155.,0.,457.])
    # put them together so we can use for loop 
    screws = np.vstack((S1,S2,S3,S4,S5,S6))
    
    # define the M matrix (the zero-position e-e transformation matrix)
    M = np.array([[1,0,0,457],
                  [0,1,0,1],
                  [0,0,1,155],
                  [0,0,0,1]])
    
    # computing the T matrix using forward kinematics
    T = np.eye(4)
    for s,t in zip(screws,theta):
        se3 = VecTose3(s,t)
        T = T@se3
    return T@M


def deg2rad(angels):
    """
    transforming vectors of angles from degree to radias 
    """
    return np.array([np.deg2rad(angels[0]),
                     np.deg2rad(angels[1]),
                     np.deg2rad(angels[2]),
                     np.deg2rad(angels[3]),
                     np.deg2rad(angels[4]),
                     np.deg2rad(angels[5])])

data1 = deg2rad(np.array([-20,-40,60,10,30,0]))
data2 = deg2rad(np.array([5,10,-30,230,-50,150]))
data3 = deg2rad(np.array([-35,-90,60,145,45,10]))


T1 = lab2(data1)
T2 = lab2(data2)
T3 = lab2(data3)
print("The transformation matix T1 is:\n",
      T1,"\n"
      "The position of end-effector is:\n ",
      T1[:3,3].T,"\n")
print("The transformation matix T2 is:\n",
      T2,"\n"
      "The position of end-effector is:\n ",
      T2[:3,3].T,"\n")
print("The transformation matix T3 is:\n",
      T3,"\n"
      "The position of end-effector is:\n ",
      T3[:3,3].T,"\n")



