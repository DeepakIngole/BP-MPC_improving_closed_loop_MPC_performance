import sys, os
import casadi as ca
import numpy as np
from typing import Tuple,List
from datetime import datetime
import pickle

# add root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.poles_to_linear_sys import poles_to_linear_sys
from examples.dynamics import random_linear

# SAMPLING TIME FOR DYNAMICS
TS = 0.15

# state dimension
NX = 4

# magnitude of (continuous time) poles
POLE_RANGE = [-5,1]

# horizons
HORIZON = 20

# uncertainty on poles
POLE_UNCERTAINTY = 1

# how spread out the initial condition is
X0_MAG = 2

# decide whether to include noise or not
NOISE_MAG = 0.1

# number of models used in robust GD
N_MODELS = 10

def multiple(
        sampling_time:float=0.15,
        n_x:int=4,
        horizon:int=20,
        n_models:int = 1,
        pole_range:Tuple[float,float]=[-5,1],
        pole_uncertainty:float=1,
        x0_mag = 2,
        noise_mag:float = 0.0,
    ) -> List[dict]:
    """
    Generates a list of random linear models with specified properties.

    This function creates `n_models` random linear models, each with `n_x` states, sampled at the specified `sampling_time`.
    The models are generated with random poles within the given `pole_range` and optional pole uncertainty. The initial state
    dispersion and process noise magnitude can also be specified.

    Args:
        sampling_time (float, optional): Sampling time for the models. Must be positive. Default is 0.15.
        n_x (int, optional): Number of states in each model. Must be positive. Default is 4.
        horizon (int, optional): Prediction horizon. Must be positive. Default is 20.
        n_models (int, optional): Number of models to generate. Must be positive. Default is 1.
        pole_range (Tuple[float, float], optional): Range [min, max] for the real part of the poles. Default is [-5, 1].
        pole_uncertainty (float, optional): Uncertainty range for the poles. Must be non-negative. Default is 1.
        x0_mag (float, optional): Magnitude of the initial state dispersion. Must be non-negative. Default is 2.
        noise_mag (float, optional): Magnitude of the process noise. Must be non-negative. Default is 0.1.

    Returns:
        List[dict]: A list of dictionaries, each representing a generated random linear model.

    Raises:
        AssertionError: If any of the input arguments do not satisfy their constraints.

    Side Effects:
        Saves the generated models to a pickle file in the './.models/' directory. The filename includes the timestamp,
        number of states, pole range, number of models, and whether noise is present.
    """

    assert sampling_time > 0, 'Sampling time must be positive.'
    assert n_x > 0, 'Number of states must be positive.'
    assert horizon > 0, 'Horizon must be positive.'
    assert n_models > 0, 'Number of models must be positive.'
    assert pole_range[0] <= pole_range[1], 'Pass pole_range as pole_range = [pole_min, pole_max] with pole_min <= pole_max.'
    assert pole_uncertainty >= 0, 'Pole uncertainty range must be non-negative.'
    assert x0_mag >= 0, 'Dispersion of initial state must be non-negative.'
    assert noise_mag >= 0, 'Noise magnitude must be non-negative.'

    # initialize model list
    model_list = []

    # generate n_models random linear models
    for i in range(n_models):
        model = single(
                sampling_time=sampling_time,
                n_x=n_x,
                noise_mag=noise_mag,
                pole_range=pole_range,
                horizon=horizon,
                pole_uncertainty=pole_uncertainty
            )
        model_list.append(model)

    return model_list

# generate several models and store
def single(
        sampling_time:float=0.15,
        n_x:int=4,
        horizon:int=20,
        pole_range:Tuple[float,float]=[-5,1],
        pole_uncertainty:float=1.0,
        noise_mag:float=0.0
    ) -> dict:
    """
    Generates a single random linear system model with optional uncertainty and noise.
    This function creates a random linear system (e.g., for control or identification experiments)
    with specified state dimension, sampling time, and pole range. It can also introduce uncertainty
    in the system poles and optionally add process noise.

    Args:
        sampling_time (float, optional): Discretization time step for the system. Defaults to 0.15.
        n_x (int, optional): Number of states in the system. Defaults to 4.
        horizon (int, optional): Time horizon for noise realization. Defaults to 20.
        pole_range (Tuple[float, float], optional): Range for sampling the true system poles. Defaults to [-5, 1].
        pole_uncertainty (float, optional): Magnitude of uncertainty to apply to the poles. Defaults to 1.0.
        noise_mag (float, optional): Magnitude of process noise. If 0, no noise is added. Defaults to 0.0.
    
    Returns:
        dict: A dictionary containing:
            - 'dyn_dict': Dictionary with the true system dynamics.
            - 'theta_uncertain': Parameter vector for the uncertain system.
            - 'theta_true': Parameter vector for the true system.
            - 'x0': Initial state vector.
            - 'w0': Realization of process noise (if noise_mag > 0), else None.
            - 'poles_true': True system poles.
            - 'poles_uncertain': Sampled uncertain poles.
            - 'A_uncertain': State matrix of the uncertain system.
            - 'B_uncertain': Input matrix of the uncertain system.
    """

    # check if noise should be used
    use_noise = noise_mag > 0

    # create dictionary with parameters of cart pendulum
    dyn_dict,true_theta,true_poles = random_linear.dynamics(Ts=sampling_time,n_x=n_x,use_w=use_noise,pole_mag=pole_range)

    # get inputs to x_next
    func_in = [ dyn_dict[key] for key in ['x','u','w'] if key in dyn_dict ]
    func_in_names = [ key for key in ['x','u','w'] if key in dyn_dict ]

    # get inputs to x_next_nom
    func_in_nom = [ dyn_dict[key] for key in ['x','u','theta'] if key in dyn_dict ]
    func_in_names_nom = [ key for key in ['x','u','theta'] if key in dyn_dict ]

    # form x_next function
    f = ca.Function('f', func_in, [dyn_dict['x_next']], func_in_names, ['x_next'])

    # form x_next_nom function
    f_nom = ca.Function('f_nom', func_in_nom, [dyn_dict['x_next_nom']], func_in_names_nom, ['x_next_nom'])

    # set initial conditions
    x0 = ca.DM.ones(n_x,1)
    # x0 = ca.DM( X0_MAG * (np.ones((n_x,1)) + 2*np.random.rand(n_x,1)) )

    # create new system with uncertainty by sampling new poles within the specified
    # uncertainty range
    poles_uncertain = pole_uncertainty*(np.ones(n_x)+2*np.random.rand(n_x))
    A_uncertain,B_uncertain,_ = poles_to_linear_sys(poles_uncertain,Ts=sampling_time)
    theta0 = ca.DM(ca.vertcat(ca.vec(A_uncertain),ca.vec(B_uncertain)))

    # sample noise if requested
    if use_noise:
        
        # get dimension of noise
        n_w = dyn_dict['w'].shape[0]

        # get random noise realization
        w0 = ca.horzsplit(noise_mag*(2*np.random.rand(n_w,horizon)-np.ones((n_w,horizon))))

    else:

        w0 = None
    
    # form output dictionary
    dyn_dict = {
        'f': f,
        'f_nom':f_nom,
        'theta_uncertain': theta0,
        'theta_true': true_theta,
        'x0': x0,
        'w0': w0,
        'poles_true': true_poles,
        'poles_uncertain': poles_uncertain,
        'A_uncertain': ca.DM(A_uncertain),
        'B_uncertain': ca.DM(B_uncertain),
    }

    return dyn_dict

if __name__ == "__main__":

    # check if .models directory exists
    if not os.path.exists('./.models/'):
        os.makedirs('./.models/')
    
    # get randomly generated linear models
    model_list = multiple(
        sampling_time=TS,
        n_x=NX,
        horizon=HORIZON,
        n_models=N_MODELS,
        pole_range=POLE_RANGE,
        pole_uncertainty=POLE_UNCERTAINTY
    )

    # get current date and time
    formatted_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # create file name containint NX, n_modelN_MODELSs and POLE_MAG and whether or not NOISE is present
    if NOISE_MAG > 0:
        file_name = './.models/' + formatted_date + f'_random_linear_models_NX_{NX}_POLE_MAG_{POLE_RANGE[0]}_to_{POLE_RANGE[1]}_N_MODELS_{N_MODELS}_NOISE.pkl'
    else:
        file_name = './.models/' + formatted_date + f'_random_linear_models_NX_{NX}_POLE_MAG_{POLE_RANGE[0]}_to_{POLE_RANGE[1]}_N_MODELS_{N_MODELS}.pkl'

    # dump model to file
    with open(file_name, 'wb') as f:
        pickle.dump(model_list, f)