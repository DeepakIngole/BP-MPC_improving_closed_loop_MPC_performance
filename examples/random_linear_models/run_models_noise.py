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
from utils.cost_utils import quad_cost_and_bounds,bound2poly,param2terminal_cost,quad_cost_2_param,dare2param
from src.upper_level import UpperLevel
from utils.parameter_update import \
    robust_gradient_descent, \
    gradient_descent, \
    gradient_descent_clipped, \
    robust_adam, \
    adam, \
    heavy_ball, \
    minibatch_descent_clipped
from utils.sys_id import rls, rls_robust
from utils.sample_utils import sample_unit_ball

# cleanup jit files
cleanup()

#### SETUP ---------------------------------------------------------------------------------------

# printout
PRINT_CST_VIOL = False
PRINT_CK = False

# decide what to compile
COMPILE_DYNAMICS = False
COMPILE_QP_SPARSE = False
COMPILE_QP_DENSE = False
COMPILE_JAC = False

# solver
SOLVER = 'daqp'

# mode of operation (nominal or robust)
MODE = 'nominal'

# number of models to simulate
N_MODELS = 10

# horizons
MPC_HORIZON = 5
ITERATIONS = 500

# penalties on constraint violation (closed-loop)
L2_PENALTY = 100
L1_PENALTY = 100
L_PROJ = 5

# system identification parameters
LAM = 0.1
DELTA = 0.01
R = 1
S = 5*1.15 # max pole magnitude times 1+Ts

# penalties on constraint violation (mpc)
MPC_S_QUAD = 15
MPC_S_LIN = 25

# update algorithm (options: 'Adam', 'gd', or 'heavy_ball')
UPDATE_ALGORITHM = 'gd'

# select a single model in the list (set to None to simulate all models)
MODEL_SELECT = None

# choose if certainty equivalence should be used
CERTAINTY_EQUIVALENCE = True

### CHOOSE ALGORITHM ----------------------------------------------------------------------------

if UPDATE_ALGORITHM == 'Adam':

    # Adam parameters
    ADAM_ALPHA = 0.15
    ADAM_EPSILON = 1e-6
    ADAM_BETA_1 = 0.5
    ADAM_BETA_2 = 0.8

    # save in dictionary
    hyper_parameters = {'alg':UPDATE_ALGORITHM, 'alpha':ADAM_ALPHA, 'epsilon':ADAM_EPSILON, 'beta_1':ADAM_BETA_1, 'beta_2':ADAM_BETA_2, 'iterations':ITERATIONS}

elif UPDATE_ALGORITHM == 'gd':

    if CERTAINTY_EQUIVALENCE:

        # gd paramers (CE)
        RHO = 0.075
        ETA = 0.8
        LOG = True
        CLIP = 150
        BATCH_SIZE = 1

    else:

        # gd paramers (no CE)
        RHO = 3e-2
        ETA = 0.51
        LOG = True
        CLIP = 300

    # save in dictionary
    hyper_parameters = {'alg':UPDATE_ALGORITHM, 'rho':RHO, 'eta':ETA, 'log':LOG, 'clip':CLIP, 'iterations':ITERATIONS}

elif UPDATE_ALGORITHM == 'heavy_ball':

    if CERTAINTY_EQUIVALENCE:

        # heavy ball parameters (CE)
        RHO = 0.25
        ETA = 0.6
        LOG = True
        BETA0 = 0.5

    else:
        
        # heavy ball parameters (no CE)
        RHO = 1e-3
        ETA = 0.6
        LOG = True
        BETA0 = 0.3

    # save in dictionary
    hyper_parameters = {'alg':UPDATE_ALGORITHM, 'rho':RHO, 'eta':ETA, 'log':LOG, 'beta0':BETA0, 'iterations':ITERATIONS}


#### GET THE MODELS -----------------------------------------------------------------------------

# load latest model from .models directory
cwd = os.path.dirname(os.path.abspath(__file__))
main_dir = '/'.join(cwd.split('/')[:-2])
models_dir = os.path.join(main_dir, '.models')
all_models = glob(models_dir + "/*.pkl")

# remove models that don't have noise
all_models_noise = [model for model in all_models if 'NOISE' in model]

assert len(all_models_noise) > 0, 'No noisy models found.'

# Extract the datetime from the string
datetimes = []
for s in all_models_noise:
    # Extract the datetime string using split
    parts = s.split('/')
    date_str = parts[-1].split('_random')[0]  # '2025_05_23_13_15_47'
    dt = datetime.strptime(date_str, '%Y_%m_%d_%H_%M_%S')
    datetimes.append(dt)

# Get the index of the most recent datetime
most_recent_index = max(range(len(datetimes)), key=lambda index: datetimes[index])

# load using pickle
with open(all_models_noise[most_recent_index], 'rb') as f:
    model_list = pickle.load(f)


#### LOOP THROUGH ALL MODELS ---------------------------------------------------------------------

ALL_MODELS = []

# we assume all models either use or don't use noise
use_noise = model_list[0]['dim']['w'] > 0

if not use_noise:
    raise Exception('This script should run with noise.')

if MODEL_SELECT is not None and not isinstance(MODEL_SELECT,list):
    MODEL_SELECT = [MODEL_SELECT]

# choose models to simulate
model_to_simulate = [model_list[i] for i in MODEL_SELECT] if MODEL_SELECT is not None else model_list

# select verbosity
simulation_verbosity = 0 if MODEL_SELECT is None else 1

# setup printout
columns = [ ("MODEL", 7)]
if PRINT_CST_VIOL:
    columns.extend([("Cst training", 25), ("Cst testing", 25)])
columns.extend([("Training Cost (Untrained-Trained-Difference)", 44),
                ("Training Best (Feedback-Omniscient)", 49),
                ("Testing Cost (Trained-Difference)", 33),
                ("Testing Best (Feedback-Omniscient)", 49),
                ("Dare cost",11),
                ('Error on identified theta', 25)])
if PRINT_CK:
    columns.append(("Uncertainty radius", 18))
columns.append(("QP failed", 10))

# Create a format string based on column widths
format_str = " | ".join(f"{{:^{width}}}" for _, width in columns)

# Print header
header = format_str.format(*(name for name, _ in columns))
separator = "-+-".join("-" * width for _, width in columns)
print(header)
print(separator)

# store string
main_printout = [header,separator]

# store results
results_list = []

# loop through all models
for i,model in enumerate(model_to_simulate):

    # dictionary to generate dynamics
    dyn_dict = {}

    # generate symbolic variables
    dyn_dict['x'] = ca.SX.sym('x',model['dim']['x'],1)
    dyn_dict['u'] = ca.SX.sym('x',model['dim']['u'],1)
    dyn_dict['theta'] = ca.SX.sym('theta',model['dim']['theta'],1)
    dyn_dict['w'] = ca.SX.sym('w',model['dim']['w'],1)
    
    # inputs to x_next
    x_next_inputs = {'x':dyn_dict['x'],'u':dyn_dict['u'],'w':dyn_dict['w']}
    
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
    w0 = model['w0']

    # extract upper level cost
    Q_true = model['cost_spec']['Q']
    R_true = model['cost_spec']['R']

    # and the constraints
    x_max = model['cost_spec']['x_max']
    x_min = model['cost_spec']['x_min']
    u_max = model['cost_spec']['u_max']
    u_min = model['cost_spec']['u_min']

    # parameter = terminal state cost and input cost
    c_qx = ca.SX.sym('c_qx', int(n_x * (n_x + 1) / 2), 1)
    c_qn = ca.SX.sym('c_q', int(n_x * (n_x + 1) / 2), 1)
    c_r = ca.SX.sym('c_r', 1, 1)

    # stage cost (state)
    Qx = [param2terminal_cost(c_qn) + 0.01 * ca.SX.eye(n_x)] * (MPC_HORIZON - 1)

    # stage cost (input)
    Ru = c_r**2 + 1e-6

    # create parameter
    if CERTAINTY_EQUIVALENCE:
        p = ca.vcat([c_qx,c_qn,c_r])
        pf = dyn_dict['theta']
    else:
        p = ca.vcat([c_qx,c_qn,c_r,dyn_dict['theta']])

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
    qp_options = {  'compile_qp_sparse':COMPILE_QP_SPARSE,
                    'compile_qp_dense':COMPILE_QP_DENSE,
                    'compile_jac':COMPILE_JAC,
                    'solver':SOLVER}

    # extract linearized dynamics at the origin
    A = dyn.A_nom(ca.DM(n_x,1),ca.DM(n_u,1),theta0)
    B = dyn.B_nom(ca.DM(n_x,1),ca.DM(n_u,1),theta0)

    if CERTAINTY_EQUIVALENCE:
        # create MPC
        mpc = QP(ingredients=ing, p=p, pf=pf, options=qp_options)
        # create upper level
        upper_level = UpperLevel(p=p, pf=pf, horizon=model['dim']['horizon'], mpc=mpc)
        # compute terminal cost initialization
        # p_init = ca.vertcat(quad_cost_2_param(Q_true), dare2param(A, B, Q_true, R_true), R_true)
        p_init = ca.vertcat(quad_cost_2_param(Q_true), quad_cost_2_param(Q_true), R_true)
    else:
        # create MPC
        mpc = QP(ingredients=ing, p=p, options=qp_options)
        # create upper level
        upper_level = UpperLevel(p=p, horizon=model['dim']['horizon'], mpc=mpc)
        # compute terminal cost initialization
        # p_init = ca.vertcat(quad_cost_2_param(Q_true), dare2param(A, B, Q_true, R_true), R_true, theta0)
        p_init = ca.vertcat(quad_cost_2_param(Q_true), quad_cost_2_param(Q_true), R_true, theta0)

    if MODE == 'robust':
        
        # create system identification algorithm
        sys_id_update, sys_id_init, _ = rls_robust(
            dynamics=dyn,
            n_models=N_MODELS,
            R=R,
            S=S,
            delta=DELTA,
            horizon=model['dim']['horizon'],
            lam=LAM,
            theta0=theta0,
            jit=False,
            idx_pf=range(theta0.shape[0]))
        
        # create parameter update algorithm
        if UPDATE_ALGORITHM == 'gd':
            parameter_update, parameter_init = robust_gradient_descent(rho=RHO, eta=ETA, n_models=N_MODELS, n_p=p.shape[0],log=LOG)
        elif UPDATE_ALGORITHM == 'Adam':
            parameter_update, parameter_init = robust_adam(n_models=N_MODELS, n_p=p.shape[0], alpha=ADAM_ALPHA,
                                                           beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2, epsilon=ADAM_EPSILON)
        else:
            raise Exception('Unknown UPDATE_ALGORITHM')

    elif MODE == 'nominal':

        # create system identification algorithm
        sys_id_update, sys_id_init, _ = rls(
            dynamics=dyn,
            horizon=model['dim']['horizon'],
            lam=LAM,
            theta0=theta0,
            jit=False,
            idx_pf=range(theta0.shape[0]))

        # create update functions
        if UPDATE_ALGORITHM == 'gd':
            # parameter_update, parameter_init = gradient_descent(rho=RHO, eta=ETA, log=LOG)
            # parameter_update, parameter_init = gradient_descent_clipped(rho=RHO, eta=ETA, log=LOG, clip=CLIP)
            parameter_update, parameter_init = minibatch_descent_clipped(rho=RHO, eta=ETA, log=LOG, clip=CLIP, batch_size=BATCH_SIZE)
        elif UPDATE_ALGORITHM == 'Adam':
            parameter_update, parameter_init = adam(alpha=ADAM_ALPHA, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2,
                                                    epsilon=ADAM_EPSILON)
        elif UPDATE_ALGORITHM == 'heavy_ball':
            parameter_update, parameter_init = heavy_ball(rho=RHO, eta=ETA, beta_0 = BETA0, log=LOG)
        else:
            raise Exception('Unknown UPDATE_ALGORITHM')

    else:
        raise Exception('Unknown MODE')

    upper_level.set_alg(
        parameter_update=parameter_update,
        parameter_init=parameter_init,
        sys_id_update=sys_id_update,
        sys_id_init=sys_id_init)
    
    # extract closed-loop variables for upper level
    x_cl = ca.vec(upper_level.param['x_cl'])
    u_cl = ca.vec(upper_level.param['u_cl'])
    c_k_cl = upper_level.param['psi_cl_c']
    theta_cl = upper_level.param['psi_cl_theta']

    # generate components of the cost
    track_cost, cst_viol_l1, cst_viol_l2 = quad_cost_and_bounds(Q_true,R_true,x_cl,u_cl,x_max,x_min)

    # form cost
    if CERTAINTY_EQUIVALENCE:
        cost = track_cost + L2_PENALTY*cst_viol_l2 + L1_PENALTY*cst_viol_l1
    else:
        # create penalty for projecting onto the confidence region
        # projection_penalty = ca.if_else(ca.norm_2(p[-theta0.shape[0]:]-theta_cl) <= c_k_cl,0,(ca.norm_2(p[-theta0.shape[0]:]-theta_cl) - c_k_cl)**2)
        projection_penalty = (p[-theta0.shape[0]:]-theta_cl).T@(p[-theta0.shape[0]:]-theta_cl)
        cost = track_cost + L2_PENALTY*cst_viol_l2 + L1_PENALTY*cst_viol_l1 + L_PROJ*projection_penalty
        # cost = L_PROJ*projection_penalty
        # track_cost = ca.SX(0)

    # create upper-level constraints
    Hx,hx,_,_ = bound2poly(x_max,x_min,u_max,u_min,model['dim']['horizon']+1)
    _,_,Hu,hu = bound2poly(x_max,x_min,u_max,u_min,model['dim']['horizon'])
    cst_viol = ca.vcat([Hx@ca.vec(x_cl)-hx,Hu@ca.vec(u_cl)-hu])

    # create auxiliary parameters taken from sys id procedure
    psi = {'c':ca.SX.sym('c_ul',1,1),'theta':ca.SX.sym('theta_ul',*theta0.shape)}
    
    # store in upper-level
    upper_level.set_cost(cost,track_cost,cst_viol)

    # create algorithm
    p = upper_level.param['p']
    j_p = upper_level.param['J_p']
    k = upper_level.param['k']

    # create scenario
    scenario = Scenario(dyn,mpc,upper_level)

    # initialize
    # init_dict = {'p':p_init,'pf':model['theta_true'],'x': x0,'theta':model['theta_true']} if CERTAINTY_EQUIVALENCE else {'p':p_init,'x': x0,'theta':theta0}
    init_dict = {'p':p_init,'pf':theta0,'x': x0,'theta':theta0} if CERTAINTY_EQUIVALENCE else {'p':p_init,'x': x0,'theta':theta0}
    init_dict['w'] = model['w0']

    # needed for compatibility
    if MODE == 'robust':

        # get initial radius of confidence region
        c_k_0 = sys_id_init()['c']

        # sample random models
        init_dict['theta'] = ca.horzsplit(
            ca.DM(init_dict['theta'] + c_k_0 * sample_unit_ball(scenario.dim['theta'], N_MODELS).T),1)

    # update options
    sim_options = {'save_memory': True, 'use_true_model': False, 'max_k': ITERATIONS,
                   'true_theta': np.array(model['theta_true']), 'verbosity': simulation_verbosity}
    if MODE == 'robust':
        sim_options['simulate_parallel_models'] = True

    # create dictionary for first simulation (keep only the first noise sample)
    init_dict_first = init_dict.copy()
    init_dict_first['w'] = model['w0'][0]

    # run first simulation
    _, _, first_qp_failed = scenario.simulate(options=sim_options,init=init_dict_first)

    # check if QP solver failed in the first iteration
    if first_qp_failed:
        qp_failed = 'First'

        # fill dummy lists for compatibility
        cost = [ca.inf,ca.inf]
        cst = [0,0]

    # if not, run the closed-loop optimization
    else:

        # get average training cost for untrained parameter
        sim_list_training_untrained,_,_,qp_failed_untrained = scenario.closed_loop(options=sim_options|{'mode':'simulate','max_k':len(model['w0'])},init=init_dict.copy())

        # # run without noise for a few iterations
        # init_dict_nominal = init_dict.copy()
        # init_dict_nominal['max_k'] = 10
        # sim_options_nominal = sim_options.copy()
        # sim_options_nominal['simulate_nominal'] = True
        # p_nominal_trained = scenario.closed_loop(options=sim_options_nominal,init=init_dict_nominal)[2]

        # # update parameter value
        # init_dict['p'] = p_nominal_trained

        # run closed loop
        sim_list,_,_,qp_failed_closed_loop = scenario.closed_loop(options=sim_options|{'random_sampling':False, 'gd_type':'sgd', 'batch_size':len(model['w0'])},init=init_dict)

        # compute cost and constraint violation improvement
        cost = [sim.cost for sim in sim_list]
        cst = [sim.cst for sim in sim_list]
        c_k = [sim.psi['c'] for sim in sim_list]
        theta_list = [sim.psi['theta'] for sim in sim_list]
        theta_last = theta_list[-1]
        theta_first = theta_list[0]
        p_list = [sim.p for sim in sim_list]

        # get average training cost for trained parameter
        init_dict_training_trained = init_dict.copy()
        init_dict_training_trained['p'] = p_list[-1]
        if CERTAINTY_EQUIVALENCE:
            init_dict_training_trained['pf'] = theta_last
        sim_list_training_trained,_,_,qp_failed_trained = scenario.closed_loop(options=sim_options|{'mode':'simulate','max_k':len(model['w0'])},init=init_dict_training_trained)

        # get untrained and trained costs
        untrained_cost = [sim.cost for sim in sim_list_training_untrained]
        untrained_cst = [sim.cst for sim in sim_list_training_untrained]
        trained_cost = [sim.cost for sim in sim_list_training_trained]
        trained_cst = [sim.cst for sim in sim_list_training_trained]
        mean_untrained_cost = np.mean(np.array(untrained_cost))
        mean_trained_cost = np.mean(np.array(trained_cost))
        max_untrained_cst = np.max(np.sum(np.maximum(np.hstack(untrained_cst),0),axis=0))
        max_trained_cst = np.max(np.sum(np.maximum(np.hstack(trained_cst),0),axis=0))

        # update qp_failed flag
        qp_failed = 'Sim' if qp_failed_closed_loop else 'Never'

        # validate solution on unseen samples
        if qp_failed == 'Never':

            # update initialization dictionary => add final parameter value and unseen noise
            init_dict_validate = init_dict.copy()
            init_dict_validate['w'] = model['w0_testing']
            init_dict_validate['p'] = p_list[-1]
            if CERTAINTY_EQUIVALENCE:
                init_dict_validate['pf'] = theta_last

            # run in simulation mode
            sim_list_validate,*_ = scenario.closed_loop(options=sim_options|{'mode':'simulate','max_k':len(model['w0'])},init=init_dict_validate)

            # get cost
            cost_validate = [sim.cost for sim in sim_list_validate]
            cst_validate = [sim.cst for sim in sim_list_validate]

            # compute mean cost
            mean_cost_validate = np.mean(np.array(cost_validate))
            max_cst_validate = np.max(np.sum(np.maximum(np.hstack(cst_validate),0),axis=0))

            # run with DARE solution and trained model
            A_trained = dyn.A_nom(ca.DM(n_x,1),ca.DM(n_u,1),theta_last)
            B_trained = dyn.B_nom(ca.DM(n_x,1),ca.DM(n_u,1),theta_last)
            p_dare = ca.vertcat(quad_cost_2_param(Q_true), dare2param(A_trained, B_trained, Q_true, R_true), R_true)
            init_dict_validate['p'] = p_dare
            sim_list_validate_dare,*_ = scenario.closed_loop(options=sim_options|{'mode':'simulate','max_k':len(model['w0'])},init=init_dict_validate)

            # get cost
            cost_dare = [sim.cost for sim in sim_list_validate_dare]
            cost_dare_mean = np.mean(np.array(cost_dare))

        else:

            mean_cost_validate = np.inf
            max_trained_cst = np.inf

        # use the best cost from the model
        best_cost_testing = np.mean(np.array(model['best_cost_testing']))
        best_cost_omni_testing = np.mean(np.array(model['best_cost_testing_omni']))
        best_cost_training = np.mean(np.array(model['best_cost']))
        best_cost_omni_training = np.mean(np.array(model['best_cost_omni']))
                                
        # setup printout strings
        dare_cost_string = '{0:0.4f}'.format(cost_dare_mean)
        cst_viol_string_training = '{0:0.4f}'.format(max_untrained_cst) + ' | ' + '{0:0.4f}'.format(max_trained_cst)
        cst_viol_string_testing = '{0:0.4f}'.format(max_cst_validate)
        training_cost_string =  '{0:0.4f}'.format(mean_untrained_cost) + ' | ' + \
                                '{0:0.4f}'.format(mean_trained_cost) + ' | ' + \
                                '{0:0.4f}'.format(mean_trained_cost - mean_untrained_cost)
        best_training_cost_string = '{0:0.4f}'.format(best_cost_training) + ' (' + \
                                    '{0:0.4f}'.format(best_cost_training-mean_trained_cost) + ') | ' + \
                                    '{0:0.4f}'.format(best_cost_omni_training) + ' (' + \
                                    '{0:0.4f}'.format(best_cost_omni_training-mean_trained_cost) + ') '
        testing_cost_string = '{0:0.4f}'.format(mean_cost_validate)
        best_testing_cost_string =  '{0:0.4f}'.format(best_cost_testing) + ' (' + \
                                    '{0:0.4f}'.format(best_cost_testing-mean_cost_validate) + ') | ' + \
                                    '{0:0.4f}'.format(best_cost_omni_testing) + ' (' + \
                                    '{0:0.4f}'.format(best_cost_omni_testing-mean_cost_validate) + ') '
        c_k_string = '{0:0.3f}'.format(c_k[1]) + ' | ' + '{0:0.3f}'.format(c_k[-1])

        if CERTAINTY_EQUIVALENCE:
            theta_error_string =    '{0:0.3f}'.format(ca.norm_2(model['theta_true']-theta_first).full().squeeze()) + ' | ' + \
                                    '{0:0.3f}'.format(ca.norm_2(model['theta_true']-theta_last).full().squeeze())
        else:
            theta_error_string =    '{0:0.3f}'.format(ca.norm_2(model['theta_true']-theta_last).full().squeeze()) + ' | ' \
                                    + '{0:0.3f}'.format(ca.norm_2(model['theta_true']-theta_first).full().squeeze()) + ' | ' \
                                    + '{0:0.3f}'.format(ca.norm_2(model['theta_true']-p_list[-1][-theta0.shape[0]:]).full().squeeze())

        # combine
        to_print = [MODEL_SELECT[i]] if MODEL_SELECT is not None else [i]
        if PRINT_CST_VIOL:
            to_print.extend([cst_viol_string_training,cst_viol_string_testing])
        to_print.extend([training_cost_string,best_training_cost_string,testing_cost_string,best_testing_cost_string,dare_cost_string,theta_error_string])
        if PRINT_CK:
            to_print.append(c_k_string)
        to_print.append(qp_failed)

        # generate printout
        new_string = format_str.format(*to_print)
        
        print(new_string)
        main_printout.append(new_string)

        # pack results
        results = {'constraint_violation_training':[max_untrained_cst,max_trained_cst,max_cst_validate],
                    'cost':cost,
                    'validation_cost':{'alg':mean_cost_validate,'best_feedback':best_cost_testing,'best_omni':best_cost_omni_testing,'dare':cost_dare_mean},
                    'training_cost':{'alg_trained':mean_trained_cost,'alg_untrained':mean_untrained_cost,'best_feedback':best_cost_training,'best_omni':best_cost_omni_training},
                    'qp_failed':qp_failed,
                    'c_k':c_k,
                    'theta':theta_list,
                    'theta_true':model['theta_true'],
                    'p':p_list
                }

        results_list.append(results)

# give a name to export
string_splits = all_models[most_recent_index].split('.models',1)
ce_string = 'CE' if CERTAINTY_EQUIVALENCE else 'NO_CE'
export_name = string_splits[0] + '.results/' + UPDATE_ALGORITHM + '_' + ce_string + '_' + string_splits[1][1:]

# add printout to the stuff to export
export = {'results':results_list,'printout':main_printout, 'hyperparameters':hyper_parameters}

# export results
if MODEL_SELECT is None:
    with open(export_name, 'wb') as handle:
        pickle.dump(export, handle, protocol=pickle.HIGHEST_PROTOCOL)