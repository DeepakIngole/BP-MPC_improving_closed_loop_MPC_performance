##### IMPORTS -------------------------------------------------------------------------------------

import sys, os
import casadi as ca
import numpy as np
from glob import glob
from datetime import datetime
import pickle
from prettytable import PrettyTable

# add root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.scenario import Scenario
from src.dynamics import Dynamics
from src.qp import QP
from src.ingredients import Ingredients
from utils.cleanup import cleanup
from utils.cost_utils import quad_cost_and_bounds,bound2poly,param2terminal_cost,dare2param
from src.upper_level import UpperLevel
from utils.parameter_update import robust_descent_solver
from utils.sys_id import get_c_k_func, get_phi
from utils.sample_utils import sample_unit_ball

import sys

def clear_last_lines(n):
    for _ in range(n):
        # Move cursor up one line
        sys.stdout.write('\x1b[1A')  # ANSI escape code to move cursor up
        # Clear the line
        sys.stdout.write('\x1b[2K')  # ANSI escape code to clear the line
    sys.stdout.flush()

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
N_MODELS = 10

# horizons
MPC_HORIZON = 10
ITERATIONS = 20

# penalties on constraint violation (closed-loop)
L2_PENALTY = 1000
L1_PENALTY = 1000

# system identification parameters
LAM = 0.01
DELTA = 0.01
R = 1

# penalties on constraint violation (mpc)
MPC_S_QUAD = 15
MPC_S_LIN = 25


#### GET THE MODELS -----------------------------------------------------------------------------

# load latest model from .models directory
all_models = glob("./.models/*.pkl")

# Extract the datetime from the string
datetimes = []
for s in all_models:
    # Extract the datetime string using split
    parts = s.split('/')
    date_str = parts[-1].split('_random')[0]  # '2025_05_23_13_15_47'
    dt = datetime.strptime(date_str, '%Y_%m_%d_%H_%M_%S')
    datetimes.append(dt)

# Get the index of the most recent datetime
most_recent_index = max(range(len(datetimes)), key=lambda i: datetimes[i])

# load using pickle
with open(all_models[most_recent_index], 'rb') as f:
    model_list = pickle.load(f)


#### LOOP THROUGH ALL MODELS ---------------------------------------------------------------------

ALL_MODELS = []

# initialize pretty table
table = PrettyTable()
table.field_names = ["MODEL", "Constraint violation (first-last)", "Cost (first-last-increment)", "Best achievable cost", "QP failed"]

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
    if not USE_CUSTOM_COST_SPEC and model['best_cost'] is not 0:

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
    c_q = ca.SX.sym('c_q',int(n_x*(n_x+1)/2),1)
    c_r = ca.SX.sym('c_r',1,1)

    # stage cost (state)
    Qx = [Q_true] * (MPC_HORIZON-1)

    # stage cost (input)
    Ru = c_r**2 + 1e-6

    # create parameter
    p = ca.vcat([c_q,c_r])
    pf = dyn_dict['theta']

    # MPC terminal cost
    Qn = param2terminal_cost(c_q) + 0.01*ca.SX.eye(n_x)

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

    # create upper level
    upper_level = UpperLevel(p=p,pf=pf,horizon=model['dim']['horizon'],mpc=mpc)

    # extract linearized dynamics at the origin
    A = dyn.A_nom(ca.DM(n_x,1),ca.DM(n_u,1),theta0)
    B = dyn.B_nom(ca.DM(n_x,1),ca.DM(n_u,1),theta0)

    # compute terminal cost initialization
    p_init = ca.vertcat(dare2param(A,B,Q_true,R_true),1e-1)#ca.vertcat(ca.DM.ones(p.shape[0]-1,1)*1e-3,1)#

    # extract closed-loop variables for upper level
    x_cl = ca.vec(upper_level.param['x_cl'])
    u_cl = ca.vec(upper_level.param['u_cl'])

    track_cost, cst_viol_l1, cst_viol_l2 = quad_cost_and_bounds(Q_true,R_true,x_cl,u_cl,x_max,x_min)

    # put together
    cost = track_cost + L2_PENALTY*cst_viol_l2 + L1_PENALTY*cst_viol_l1

    # create upper-level constraints
    Hx,hx,_,_ = bound2poly(x_max,x_min,u_max,u_min,model['dim']['horizon']+1)
    _,_,Hu,hu = bound2poly(x_max,x_min,u_max,u_min,model['dim']['horizon'])
    cst_viol = ca.vcat([Hx@ca.vec(x_cl)-hx,Hu@ca.vec(u_cl)-hu])

    # store in upper-level
    upper_level.set_cost(cost,track_cost,cst_viol)

    # create scenario
    scenario = Scenario(dyn,mpc,upper_level)

    # initialize
    init_dict = {'p':p_init,'pf':theta0,'x': x0,'theta':theta0}
    
    if use_noise:
        init_dict['w'] = model['w0']

    # run nominal version
    sim_options = {'save_memory':True,'use_true_model':False,'max_k':ITERATIONS,'true_theta':np.array(model['theta_true']),'verbosity':0}
    sim_nominal, _, qp_failed_nominal = scenario.simulate(options=sim_options,init=init_dict)

    # run robust version
    sim_robust, _, qp_failed_robust = scenario.simulate(options=sim_options,init=init_dict)

    # get phi function
    phi = get_phi(dynamics=dyn,horizon=model['dim']['horizon'])

    # compute feature vectors
    phi_k = np.array(phi(sim_robust.x[:,:-1],sim_robust.u))

    # compute output vector
    z_k = np.array(sim_robust.x[:,1:])

    # reshape
    phi_reshaped = phi_k.reshape(n_x,scenario.dim['theta'],model['dim']['horizon'],order='F').transpose(2,1,0)

    # update a and b
    a_k = ca.DM.eye(scenario.dim['theta'])*LAM + ca.DM(np.einsum('nij,njk->ik', phi_reshaped, phi_reshaped.transpose(0,2,1)))

    # compute confidence bound
    c_k = get_c_k_func(R=R,n_theta=scenario.dim['theta'],lam=LAM,delta=DELTA,S=ca.norm2(model['theta_true']-theta0))(np.linalg.svdvals(np.array(a_k))[-1])
    
    # run robust version
    init_dict['theta'] = ca.DM(theta0 + c_k*sample_unit_ball(scenario.dim['theta'],N_MODELS).T)
    sim_options['simulate_parallel_models'] = True

    

    if qp_failed_nominal:
        assert qp_failed_robust, 'qp should have failed!'
        qp_failed = 'True'
    else:
        qp_failed = 'False'

    # compute cosine similarities


    # add to table
    table.add_row([i, f'{ca.sum1(ca.fmax(cst[0],0))} | {ca.sum1(ca.fmax(cst[-1],0))}', f'{cost[0]} | {cost[-1]} | {cost[-1]-cost[0]}', f'{best_cost} ({best_cost-cost[-1]})', qp_failed])

    # clear previous rows
    if i > 0:
        clear_last_lines(4+i)
    
    # Print the table
    print(table)

# save table with results
print('me')

# all_models[most_recent_index]

# # dump model to file
# with open(file_name, 'wb') as f:
#     pickle.dump(model_list, f)

# while True:

#     # get cost
#     cost,track_cost,cst_viol = scenario.upper_level.cost(sim)

#     # run a single update
#     sim.j_p = scenario._mapped['j_cost'](sim)