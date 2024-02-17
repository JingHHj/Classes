import numpy as np

        # given information
time_interval = 0.5
dimeter = 0.254
tick = 360.
count = 200.

# defining given angular velocity
w_t = 1.
# defining given measurement of current time step
z_t = np.array([-0.29, 1.25])
# defining given state of current time step
x_t = np.array([0.,0.,-np.pi/18.])
# defining given measurement of next time step
z_t1 = np.array([-0.86, 0.98])

def mm_diff_drive(v,omega,tau,state):
    """
    aim: Defining the motion model of a differential drive
    arg:
        v: linear velocity at time t (number)
        omega: angular velocity at time t 
                generally its a 3*1 vector under 3-D surcumstance, 
                but for this question since its 2-D case there's only one angular velocity which is about the z-axis,
                then its a number
        tau: time interval between time step t and time step t+1
        state: the current state (3*1 vector)

    return: the state of next time step (3*1 vector)
    """
    theta = state[2]
    matrix = [v*np.sinc(omega*tau/2)*np.cos(theta+omega*tau/2),
              v*np.sinc(omega*tau/2)*np.sin(theta+omega*tau/2),
              omega]
    return state + np.dot(tau,matrix)

def v_encoder(tau,z,d,n):
    """
    aim: Defining the observation model of a encoder
    arg:
        tau: time interval between time step t and time step t+1 (number)
        z: encoder count (number)
        d: diameter of the wheel (number)
        n: ticks (number)

    return: the linear velocity of current time step (number)
    """
    return np.pi*d*z/n/tau


def find_points(state_t,measurement_t,state_t1,measurement_t1):
    """
    aim: finding out the unknown points
    arg:
        state_t: current time step (3*1 vector)
        state_t1: next time step (3*1 vector)
        measurement_t: measurement of current time step (3*1 vector)
        measurement_t1: measurement of next time step (3*1 vector)
    return: the two unknown points m and n (two 1*2 vector)
    """
    x_t, y_t, c_t = state_t
    x_t1, y_t1, c_t1 = state_t1
    a_t, b_t = measurement_t
    a_t1, b_t1 = measurement_t1

    m_x = (x_t*np.tan(a_t+c_t)-x_t1*np.tan(a_t1+c_t1)-y_t+y_t1)/(np.tan(a_t+c_t)-np.tan(a_t1+c_t1))
    m_y = (y_t/np.tan(a_t+c_t)-y_t1/np.tan(a_t1+c_t1)-x_t+x_t1)/(1/np.tan(a_t+c_t)-1/np.tan(a_t1+c_t1))

    n_x = (x_t*np.tan(b_t+c_t)-x_t1*np.tan(b_t1+c_t1)-y_t+y_t1)/(np.tan(b_t+c_t)-np.tan(b_t1+c_t1))
    n_y = (y_t/np.tan(b_t+c_t)-y_t1/np.tan(b_t1+c_t1)-x_t+x_t1)/(1/np.tan(b_t+c_t)-1/np.tan(b_t1+c_t1))

    return np.array([m_x,m_y]), np.array([n_x,n_y])


veloctiy = v_encoder(time_interval,count,dimeter,tick)
x_t1 = mm_diff_drive(veloctiy,w_t,time_interval,x_t)
m,n = find_points(x_t,z_t,x_t1,z_t1)

print("robot state x_t+1 at time t+1: ",x_t1)
print("unknown point m: ",m)
print("unknown point m: ",n)


