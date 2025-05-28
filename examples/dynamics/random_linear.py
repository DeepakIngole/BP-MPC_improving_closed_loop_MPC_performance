import casadi as ca
import numpy as np
from typing import Tuple
import sys, os

# add root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.linear import poles_to_linear_sys

def dynamics(
        Ts:float=0.1,
        n_x:int=2,
        pole_mag:Tuple[float,float]=[0.5,1.2],
        use_theta:bool=True,
        use_w:bool=True,
        verbose=False
    ) -> dict:
    """
    Generates a random discrete-time linear system with configurable state dimension, pole range, and optional symbolic parameters.
    Args:
        Ts (float, optional): Sampling time for discretization. Defaults to 0.1.
        n_x (int, optional): Number of states. Defaults to 2.
        pole_mag (Tuple[float, float], optional): Range [min, max] for randomly generated continuous-time poles. Defaults to [0.5, 1.2].
        use_theta (bool, optional): If True, creates symbolic parameters for the nominal model (A, B matrices). Defaults to True.
        use_w (bool, optional): If True, includes process noise as a symbolic variable. Defaults to True.
        verbose (bool, optional): If True, prints generated poles and eigenvalues. Defaults to False.
    Returns:
        out (dict): Dictionary containing symbolic variables and system dynamics:
            - 'x': State symbolic variable (n_x x 1).
            - 'u': Input symbolic variable (1 x 1).
            - 'theta': (optional) Symbolic parameter vector for nominal model (if use_theta=True).
            - 'w': (optional) Symbolic process noise (if use_w=True).
            - 'x_next': Symbolic expression for successor state (true dynamics).
            - 'x_next_nom': Symbolic expression for nominal successor state.
        true_theta (ca.DM): True parameter vector (flattened Jacobian of x_next w.r.t. [x; u]).
        poles (np.ndarray): Array of generated continuous-time poles.
    Raises:
        AssertionError: If pole_mag is not in the form [min, max] with min <= max.
    Note:
        Requires CasADi (as `ca`) and NumPy (as `np`). The helper function `poles_to_linear_sys` must be defined elsewhere.
    """

    # always one input
    n_u = 1

    # define state and input symbolic variables
    x = ca.SX.sym('x',n_x,1)
    u = ca.SX.sym('u',n_u,1)

    assert pole_mag[0] <= pole_mag[1], 'Pass pole_mag as pole_mag = [pole_min, pole_max] with pole_min <= pole_max.'

    # generate random continuous-time poles
    poles = np.random.rand(n_x)*(pole_mag[1]-pole_mag[0]) + np.ones(n_x)*pole_mag[0]

    # generate random discrete-time system
    A,B,eig_A = poles_to_linear_sys(poles=poles,sampling_time=Ts)

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