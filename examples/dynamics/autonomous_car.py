import casadi as ca
from typing import Union
# from scipy.linalg import expm
import numpy as np

def dynamics(uncertainty:Union[ca.SX,ca.DM]=ca.SX.zeros(8)) -> dict:

    # define dimensions of the problem
    n_x = 4               # number of states
    n_u = 1               # number of inputs
    n_theta = 8           # number of unknown parameters
    n_w = 2               # number of disturbances

    # define symbolic variables
    x = ca.SX.sym('x0',n_x,1)
    u = ca.SX.sym('u0',n_u,1)
    theta = ca.SX.sym('theta',n_theta,1)
    w = ca.SX.sym('w',n_w,1)

    # exact parameters
    c_f, c_r, m, v_x, l_f, l_r, i_z = 155494.663, 155494.663, 1140.0, 5.0, 1.165, 1.165, 1436.24

    # sampling time
    delta_t = 0.01

    # true A matrix
    a_mat = np.zeros((4,4))
    a_mat[0,1] = 1
    a_mat[1,1] = -(c_f+c_r)/(m*v_x)
    a_mat[1,2] = (c_f+c_r)/m
    a_mat[1,3] = (c_f*l_f-c_r*l_f)/(m*v_x)
    a_mat[2,3] = 1
    a_mat[3,1] = (c_r*l_r-c_f*l_f)/(i_z*v_x)
    a_mat[3,2] = (c_f*l_f-c_r*l_r)/(i_z)
    a_mat[3,3] = -(c_f*l_f**2+c_r*l_r**2)/(i_z*v_x)

    # nominal A matrix
    a_mat_nom = ca.SX(4,4)
    a_mat_nom[0,1] = 1
    a_mat_nom[1,1] = theta[0]
    a_mat_nom[1,2] = theta[1]
    a_mat_nom[1,3] = theta[2]
    a_mat_nom[2,3] = 1
    a_mat_nom[3,1] = theta[3]
    a_mat_nom[3,2] = theta[4]
    a_mat_nom[3,3] = theta[5]

    # true B matrix
    b_mat = ca.SX(4,1)
    b_mat[1,0] = c_f/m
    b_mat[3,0] = l_f*c_f/i_z

    # nominal B matrix
    b_mat_nom = ca.SX(4,1)
    b_mat_nom[1,0] = theta[6]
    b_mat_nom[3,0] = theta[7]

    # true disturbance matrix
    w_mat = ca.SX(n_x,n_w)
    w_mat[1,0] = 1
    w_mat[3,1] = 1

    # exact discretization
    # a_mat_disc = ca.DM(expm(a_mat*delta_t))
    # temp_mat = ca.SX(ca.solve(ca.DM(a_mat), a_mat_disc - ca.DM.eye(n_x)))
    # b_mat_disc = temp_mat @ b_mat
    # w_mat_disc = temp_mat @ w_mat
    a_mat_disc = a_mat*delta_t + ca.SX.eye(n_x)
    b_mat_disc = b_mat*delta_t
    w_mat_disc = w_mat*delta_t

    # Euler discretization for nominal model
    a_mat_disc_nom = a_mat_nom*delta_t + ca.SX.eye(n_x)
    b_mat_disc_nom = b_mat_nom*delta_t

    # compute next state symbolically
    x_next = ca.SX(a_mat_disc) @ x + b_mat_disc @ u + w_mat_disc @ w

    # create nominal model
    x_next_nom = a_mat_disc_nom @ x + b_mat_disc_nom @ u

    # construct nominal theta
    true_theta = ca.vertcat(a_mat[1,1:4], a_mat[3,1:4], b_mat[1,0], b_mat[3,0])

    # add uncertainty
    nominal_theta = ca.diag(uncertainty+ca.SX.ones(n_theta)) @ true_theta

    # output dictionary
    out = {'x':x, 'u':u, 'theta':theta, 'w':w, 'x_next':x_next, 'x_next_nom':x_next_nom}

    # check that parameters are correct
    x_next_euler = (a_mat*delta_t+ca.SX.eye(n_x))@x+b_mat*delta_t@u
    input_args = {'x':ca.DM(np.random.rand(n_x,1)), 'u':ca.DM(np.random.rand(n_u,1)), 'theta':true_theta}
    error_should_be_zero = ca.Function('check_params',[x,u,theta],[x_next_nom-x_next_euler],['x','u','theta'],['error']).call(input_args)['error']
    assert ca.mmax(ca.fabs(ca.DM(error_should_be_zero))) == 0, 'True theta is not correct.'
    if ca.mmax(ca.fabs(uncertainty-ca.SX.zeros(8))) > 0:
        input_args['theta'] = nominal_theta
        error_should_not_be_zero = ca.Function('check_params',[x,u,theta],[x_next_nom-x_next_euler],['x','u','theta'],['error']).call(input_args)['error']
        assert ca.mmax(ca.fabs(ca.DM(error_should_not_be_zero))) > 0, 'Nominal theta is not correct.'

    return out, true_theta, nominal_theta