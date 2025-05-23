import casadi as ca
import numpy as np
from utils.poles_to_linear_sys import poles_to_linear_sys
from typing import Tuple

def dynamics(Ts:float=0.1,n_x:int=2,pole_mag:Tuple[float,float]=[0.5,1.2],use_theta:bool=True,use_w:bool=True,verbose=False) -> dict:

    # always one input
    n_u = 1

    # define state and input symbolic variables
    x = ca.SX.sym('x',n_x,1)
    u = ca.SX.sym('u',n_u,1)

    assert pole_mag[0] <= pole_mag[1], 'Pass pole_mag as pole_mag = [pole_min, pole_max] with pole_min <= pole_max.'

    # generate random continuous-time poles
    poles = np.random.rand(n_x)*(pole_mag[1]-pole_mag[0]) + np.ones(n_x)*pole_mag[0]

    # generate random discrete-time system
    A,B,eig_A = poles_to_linear_sys(poles=poles,Ts=Ts)

    if verbose:
        print(f'Generated poles: {poles}')
        print(f'Generated eigenvalues: {eig_A}')

    # create output dictionary
    out = {'x':x, 'u':u}
    
    # nominal model is the entire A and B matrices
    if use_theta:

        # create symbolic variable
        theta = ca.SX.sym('theta',n_x*(n_x+n_u),1)

        # append to dictionaries
        out['theta'] = theta

        # create nominal dynamics
        A_nom = ca.reshape(theta[:n_x*n_x],n_x,n_x)
        B_nom = ca.reshape(theta[n_x*n_x:],n_x,n_u)

    # otherwise nominal and true models coincide
    else:
        A_nom = A
        B_nom = B

    # create successor state and nominal successor state
    x_next = ca.cse(ca.sparsify(A@x + B@u))
    x_next_nom = ca.cse(ca.sparsify(A_nom@x + B_nom@u))

    # print true theta
    true_theta = ca.DM(ca.vec(ca.jacobian(x_next,ca.vertcat(x,u))))

    # noise if required
    if use_w:

        # generate w
        w = ca.SX.sym('w',n_x,1)
        out['w'] = w

        # add to true dynamics
        x_next = x_next + w

    # add to dictionaries
    out['x_next'] = x_next
    out['x_next_nom'] = x_next_nom

    return out,true_theta,poles