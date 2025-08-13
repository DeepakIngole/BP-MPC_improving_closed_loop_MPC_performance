import sys
import os
import casadi as ca
from numpy.random import randint, rand
import datetime

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dynamics import Dynamics
from src.scenario import Scenario
from src.qp import QP
from src.ingredients import Ingredients
from utils.sample_elements import sample_dynamics, sample_ingredients, sample_upper_level

def test_set_init_and_set_options():

    # sample elemnts at random
    mpc_horizon = randint(1,5)
    upper_horizon = randint(2,7)
    n_models = randint(3,5)

    # generate noise free dynamics
    dynamics_dict, _ = sample_dynamics(use_d=False,use_w=False,use_theta=True,nonlinear=False)

    # extract theta
    theta = dynamics_dict['theta']

    # create dynamics
    dynamics = Dynamics(dynamics_dict)
    
    # create ingredients
    p, _, cost, constraints = sample_ingredients(dynamics.dim,p=True,horizon=mpc_horizon)
    ingredients = Ingredients(horizon=mpc_horizon,cost=cost,constraints=constraints,dynamics=dynamics)

    # create MPC
    mpc = QP(ingredients=ingredients,p=p,pf=theta)

    # create sample upper level
    upper_level = sample_upper_level(p=p,pf=theta,mpc=mpc,horizon=upper_horizon)

    # create scenario
    scenario = Scenario(dyn=dynamics,mpc=mpc,upper_level=upper_level)

    # get dimensions for simplicity
    n = scenario.dim

    # generate some thetas
    theta0 = ca.horzsplit(rand(n['theta'],n_models))

    # initialize
    init_dict_1 = {'p':ca.DM(rand(n['p'],1)),'x': ca.DM(rand(n['x'],1)),'u': ca.DM(rand(n['u'],1)),'theta':theta0,'pf':theta0[0]}
    init_dict_2 = {'p':ca.DM(rand(n['p'],1)),'x': ca.DM(rand(n['x'],1)),'u': ca.DM(rand(n['u'],1)),'theta':theta0,'pf':theta0[0]}

    # initialize options
    options_1 = {'shift_linearization': False, 'warmstart_first_qp': False, 'warmstart_shift': False,
                'epsilon': 1e-5, 'roundoff_qp': 11, 'mode': 'optimize', 'gd_type': 'gd',
                'figures': True, 'random_sampling': True, 'debug_qp': True,
                'compute_qp_ingredients': True, 'verbosity': 0, 'max_k': 250,
                'use_true_model': False, 'simulate_parallel_models': True,
                'compile_mapped_dynamics':False}
    options_2 = {'shift_linearization': True, 'warmstart_first_qp': True, 'warmstart_shift': True,
                'epsilon': 1e-6, 'roundoff_qp': 10, 'mode': 'optimize', 'gd_type': 'gd',
                'figures': False, 'random_sampling': False, 'debug_qp': False,
                'compute_qp_ingredients': False, 'verbosity': 1, 'max_k': 200,
                'use_true_model': True, 'simulate_parallel_models': False,
                'compile_mapped_dynamics':False}
    
    # set init_dict_1 as initialization
    scenario.set_init(init_dict_1)

    # set options
    scenario.update_options(options_1)

    # simulate as is
    sim,*_ = scenario.simulate()

    # check that all parameters are set correctly
    assert ca.logic_all(sim.p == init_dict_1['p']) and ca.logic_all(sim.x[:,0] == init_dict_1['x']) and ca.logic_all(sim.theta == ca.hcat(init_dict_1['theta'])) and ca.logic_all(sim.pf == init_dict_1['pf']), 'Parameters in first dictionary do not match'

    # check that options match
    assert all( [ val == scenario.options[key] for key,val in options_1.items() ] ), 'Options 1 was not properly set.'

    # simulate with other initialization dictionary and other options
    sim2,*_ = scenario.simulate(init=init_dict_2,options=options_2)

    # check that init has not been changed
    assert ca.logic_all(scenario.init['p'] == init_dict_1['p']) and ca.logic_all(scenario.init['x'] == init_dict_1['x']) and ca.logic_all(ca.hcat(scenario.init['theta']) == ca.hcat(init_dict_1['theta'])) and ca.logic_all(scenario.init['pf'] == init_dict_1['pf']), 'Initialization was mistakenly changed.'
    assert all( [ val == scenario.options[key] for key,val in options_1.items() ] ), 'Options were mistakenly changed.'

    # check that second simulation used correct parameters
    assert ca.logic_all(sim2.p == init_dict_2['p']) and ca.logic_all(sim2.x[:,0] == init_dict_2['x']) and ca.logic_all(sim2.theta == ca.hcat(init_dict_2['theta'])) and ca.logic_all(sim2.pf == init_dict_2['pf']), 'Parameters in second dictionary do not match'

if __name__ == '__main__':
    test_set_init_and_set_options()