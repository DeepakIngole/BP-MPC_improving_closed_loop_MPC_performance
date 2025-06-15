##### IMPORTS -------------------------------------------------------------------------------------

import sys, os
import casadi as ca
import numpy as np
from glob import glob
from datetime import datetime
import pickle

# add root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.scenario import Scenario
from src.dynamics import Dynamics
from src.qp import QP
from src.ingredients import Ingredients
from utils.cleanup import cleanup
from utils.cost_utils import quad_cost_and_bounds, bound2poly, param2terminal_cost, dare2param, quad_cost_2_param
from src.upper_level import UpperLevel
from utils.parameter_update import get_robust_descent_solver
from utils.sample_utils import sample_unit_ball
from utils.sys_id import get_c_k_func

# cleanup jit files
cleanup()

#### SETUP ---------------------------------------------------------------------------------------

# decide if the cost specifications contained in model list should be
# overwritten by the user-defined cost specifications
USE_CUSTOM_COST_SPEC = False

# if USE_CUSTOM_COST_SPEC = True, then use the following to specify the cost
COST_SPEC = {'Q':10.0, 'R':1.0, 'x_max':5.0, 'x_min':-5.0, 'u_max':1.0, 'u_min':-1.0}

# decide what to compile
COMPILE_DYNAMICS = False
COMPILE_QP_SPARSE = False
COMPILE_QP_DENSE = False
COMPILE_JAC = False

# solver
SOLVER = 'daqp'

# mode of operation (nominal or robust)
MODE = 'robust'

# number of models to simulate
N_MODELS = 200

# horizons
MPC_HORIZON = 5
ITERATIONS = 20

# penalties on constraint violation (closed-loop)
L2_PENALTY = 10
L1_PENALTY = 10

# system identification parameters
LAM = 1
DELTA = 0.01
R = 1e-1

# penalties on constraint violation (mpc)
MPC_S_QUAD = 15
MPC_S_LIN = 25


#### GET THE MODELS -----------------------------------------------------------------------------

# load latest model from .models directory
cwd = os.path.dirname(os.path.abspath(__file__))
main_dir = '/'.join(cwd.split('/')[:-2])
models_dir = os.path.join(main_dir, '.models')
all_models = glob(models_dir + "/*.pkl")

# Extract the datetime from the string
datetimes = []
for s in all_models:
    # Extract the datetime string using split
    parts = s.split('/')
    date_str = parts[-1].split('_random')[0]  # '2025_05_23_13_15_47'
    dt = datetime.strptime(date_str, '%Y_%m_%d_%H_%M_%S')
    datetimes.append(dt)

# Get the index of the most recent datetime
most_recent_index = max(range(len(datetimes)), key=lambda index: datetimes[index])

# load using pickle
with open(all_models[most_recent_index], 'rb') as f:
    model_list = pickle.load(f)


#### LOOP THROUGH ALL MODELS ---------------------------------------------------------------------

ALL_MODELS = []

# setup printout
columns = [
    ("MODEL", 10),
    ("QP failed", 10),
    ("Cosine similarity nominal", 25),
    ("Cosine similarity robust", 25),
    ("Relative error nominal", 22),
    ("Relative error robust", 22),
    ('c_k', 10)
]

# Create a format string based on column widths
format_str = " | ".join(f"{{:^{width}}}" for _, width in columns)

# Print header
header = format_str.format(*(name for name, _ in columns))
separator = "-+-".join("-" * width for _, width in columns)
print(header)
print(separator)

# loop through all models
for i,model in enumerate(model_list):

    # check if model uses noise
    use_noise = model['dim']['w'] > 0

    # dictionary to generate dynamics
    dyn_dict = {}

    # generate symbolic variables
    dyn_dict['x'] = ca.SX.sym('x',model['dim']['x'],1)
    dyn_dict['u'] = ca.SX.sym('x',model['dim']['u'],1)
    dyn_dict['theta'] = ca.SX.sym('theta',model['dim']['theta'],1)
    if use_noise:
        dyn_dict['w'] = ca.SX.sym('w',model['dim']['w'],1)

    # inputs to x_next
    if use_noise:
        x_next_inputs = {'x':dyn_dict['x'],'u':dyn_dict['u'],'w':dyn_dict['w']}
    else:
        x_next_inputs = {'x':dyn_dict['x'],'u':dyn_dict['u']}
    
    # inputs to x_next_nom
    x_next_nom_inputs = {'x':dyn_dict['x'],'u':dyn_dict['u'],'theta':dyn_dict['theta']}

    # generate x_next and x_next_nom
    dyn_dict['x_next'] = model['f'].call(x_next_inputs)['x_next']
    dyn_dict['x_next_nom'] = model['f_nom'].call(x_next_nom_inputs)['x_next_nom']

    # create dynamics object
    dyn = Dynamics(dyn_dict,jit=COMPILE_DYNAMICS)

    # get state and input dimensions
    n_x, n_u = dyn.dim['x'], dyn.dim['u']

    # get initializations
    x0 = model['x0']
    theta0 = model['theta_uncertain']
    if use_noise:
        w0 = model['w0']

    # check if upper level cost should be taken from model list
    if not USE_CUSTOM_COST_SPEC and model['best_cost'] != 0:

        # if so, extract upper level cost
        Q_true = model['cost_spec']['Q']
        R_true = model['cost_spec']['R']

        # and the constraints
        x_max = model['cost_spec']['x_max']
        x_min = model['cost_spec']['x_min']
        u_max = model['cost_spec']['u_max']
        u_min = model['cost_spec']['u_min']

    # otherwise, use the specifications provided
    else:

        # cost
        Q_true = COST_SPEC['Q']*ca.DM.eye(n_x)
        R_true = COST_SPEC['R']

        # constraints are simple bounds on state and input
        x_max = COST_SPEC['x_max']*ca.DM.ones(n_x,1)
        x_min = COST_SPEC['x_min']*ca.DM.ones(n_x,1)
        u_max = COST_SPEC['u_max']
        u_min = COST_SPEC['u_min']

    # parameter = terminal state cost and input cost
    c_qx = ca.SX.sym('c_qx',int(n_x*(n_x+1)/2),1) 
    c_qn = ca.SX.sym('c_q',int(n_x*(n_x+1)/2),1)
    c_r = ca.SX.sym('c_r',1,1)

    # stage cost (state)
    Qx = [param2terminal_cost(c_qn) + 0.01*ca.SX.eye(n_x)] * (MPC_HORIZON-1)

    # stage cost (input)
    Ru = c_r**2 + 1e-6

    # create parameter
    p = ca.vcat([c_qx,c_qn,c_r])
    pf = dyn_dict['theta']
    # p = ca.vcat([c_qx,c_qn,c_r,dyn_dict['theta']])

    # MPC terminal cost
    Qn = param2terminal_cost(c_qn) + 0.01*ca.SX.eye(n_x)

    # append to Qx
    Qx.append(Qn)

    # add to mpc dictionary
    cost = {'Qx': Qx, 'Ru':Ru, 's_quad':MPC_S_QUAD, 's_lin':MPC_S_LIN}

    # turn bounds into polyhedral constraints
    Hx,hx,Hu,hu = bound2poly(x_max,x_min,u_max,u_min)

    # add to mpc dictionary
    cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu, 'Hx_e':ca.SX.eye(hx.shape[0])}

    # create QP ingredients
    ing = Ingredients(horizon=MPC_HORIZON,dynamics=dyn,cost=cost,constraints=cst)

    # create options
    qp_options = {'compile_qp_sparse':COMPILE_QP_SPARSE,
                'compile_qp_dense':COMPILE_QP_DENSE,
                'compile_jac':COMPILE_JAC,
                'solver':SOLVER}

    # create MPC
    mpc = QP(ingredients=ing,p=p,pf=pf,options=qp_options)
    # mpc = QP(ingredients=ing,p=p,options=qp_options)

    # create upper level
    upper_level = UpperLevel(p=p,pf=pf,horizon=model['dim']['horizon'],mpc=mpc)
    # upper_level = UpperLevel(p=p,horizon=model['dim']['horizon'],mpc=mpc)

    # extract linearized dynamics at the origin
    A = dyn.A_nom(ca.DM(n_x,1),ca.DM(n_u,1),theta0)
    B = dyn.B_nom(ca.DM(n_x,1),ca.DM(n_u,1),theta0)

    # compute terminal cost initialization
    p_init = ca.vertcat(quad_cost_2_param(Q_true),quad_cost_2_param(Q_true),R_true)
    # p_init = ca.vertcat(quad_cost_2_param(Q_true),quad_cost_2_param(Q_true),R_true,theta0)

    # extract closed-loop variables for upper level
    x_cl = ca.vec(upper_level.param['x_cl'])
    u_cl = ca.vec(upper_level.param['u_cl'])

    track_cost, cst_viol_l1, cst_viol_l2 = quad_cost_and_bounds(Q_true,R_true,x_cl,u_cl,x_max,x_min)

    # put together
    cost = track_cost #+ L2_PENALTY*cst_viol_l2 + L1_PENALTY*cst_viol_l1

    # create upper-level constraints
    Hx,hx,_,_ = bound2poly(x_max,x_min,u_max,u_min,model['dim']['horizon']+1)
    _,_,Hu,hu = bound2poly(x_max,x_min,u_max,u_min,model['dim']['horizon'])
    cst_viol = ca.vcat([Hx@ca.vec(x_cl)-hx,Hu@ca.vec(u_cl)-hu])

    # store in upper-level
    upper_level.set_cost(cost,track_cost,cst_viol)

    # create scenario
    scenario = Scenario(dyn,mpc,upper_level)

    # construct simulation options
    sim_options = {'simulate_parallel_models': True, 'save_memory': True, 'use_true_model': False, 'max_k': ITERATIONS,
                   'true_theta': np.array(model['theta_true']), 'verbosity': 0}

    # compute confidence bound
    # c_k = get_c_k_func(R=R, n_theta=scenario.dim['theta'], lam=LAM, delta=DELTA,
    #                    S=ca.norm_2(model['theta_true'] - theta0))(LAM)
    c_k = LAM*np.linalg.norm(model['theta_true'] - theta0, ord=np.inf)

    # initialize theta: first add randomized values
    theta_init_random = ca.DM(theta0 + c_k * sample_unit_ball(scenario.dim['theta'], N_MODELS).T)

    # add nominal model and true model
    theta_init = ca.horzcat(theta_init_random,theta0,model['theta_true'])

    # initialize
    init_dict = {'p': p_init, 'pf': theta0, 'x': x0, 'theta': theta_init}
    if use_noise:
        init_dict['w'] = model['w0']

    # run robust version
    sim, _, qp_failed = scenario.simulate(options=sim_options,init=init_dict)

    # compute gradient
    j_p = scenario.upper_level.j_cost(sim)

    # get robust descent solver
    robust_descent_solver = get_robust_descent_solver(n_models=N_MODELS, n_p=scenario.dim['p'])

    # extract gradients
    j_p_list = ca.horzsplit(j_p,1)
    j_x_list = ca.horzsplit(sim.j_x,1)

    # get robust gradient
    _,j_p_robust = robust_descent_solver(ca.hcat(j_p_list[:-2]))
    j_p_average = ca.DM(np.mean(np.array(ca.hcat(j_p_list[:-2])),axis=1))

    # get nominal gradient
    j_p_nominal = np.array(j_p_list[-2]).squeeze()
    j_x_nominal = np.array(j_x_list[-2]).squeeze()

    # get true gradient
    j_p_true = np.array(j_p_list[-1]).squeeze()
    j_x_true = np.array(j_x_list[-1]).squeeze()

    # compute relative errors
    def relative_error(v,w):
        return (np.linalg.norm(v-w) / np.linalg.norm(v))
    rel_err_nominal = relative_error(j_p_true,j_p_nominal)
    rel_err_robust = relative_error(j_p_true,np.array(j_p_robust).squeeze())

    # compute cosine similarities
    def cosine_similarity(v,w):
        return np.dot(v,w) / ( np.linalg.norm(v) * np.linalg.norm(w) )
    cos_nominal = cosine_similarity(j_p_true,j_p_nominal)
    cos_robust = cosine_similarity(j_p_true, np.array(j_p_robust).squeeze())

    # Print formatted row
    print(format_str.format(i, str(qp_failed), cos_nominal, cos_robust, rel_err_nominal, rel_err_robust, str(c_k)))