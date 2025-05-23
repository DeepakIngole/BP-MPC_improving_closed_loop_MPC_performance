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
from utils.cost_utils import quad_cost_and_bounds,bound2poly,param2terminal_cost,dare2param
import examples.dynamics.random_linear as random_linear
from src.plotter import Plotter
from src.upper_level import UpperLevel
from utils.parameter_update import average_gradient_descent, robust_gradient_descent, gradient_descent
from utils.sys_id import rls, rls_robust
from utils.poles_to_linear_sys import poles_to_linear_sys
from generate_models import multiple as sample_multiple_linear_models

# cleanup jit files
cleanup()

# decide what to compile
COMPILE_DYNAMICS = False
COMPILE_QP_SPARSE = False
COMPILE_QP_DENSE = False
COMPILE_JAC = False

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
print("Index of most recent file:", most_recent_index)

# load using pickle
with open(all_models[most_recent_index], 'rb') as f:
    model_list = pickle.load(f)

# loop through all models
for i,model in enumerate(model_list):




# create dictionary with parameters of cart pendulum
dyn_dict,true_theta,true_poles = random_linear.dynamics(Ts=TS,n_x=NX,use_w=NOISE,pole_mag=POLE_MAG,verbose=True)

# model uncertainty parameter
theta = dyn_dict['theta']

# create dynamics object
dyn = Dynamics(dyn_dict,jit=COMPILE_DYNAMICS)

# get state and input dimensions
n_x, n_u = dyn.dim['x'], dyn.dim['u']

# set initial conditions
x0 = ca.DM.ones(n_x,1)
# x0 = ca.DM( X0_MAG * (np.ones((n_x,1)) + 2*np.random.rand(n_x,1)) )

# create new system with uncertainty by sampling new poles within the specified
# uncertainty range
poles_uncertain = THETA_UNCERTAINTY_RANGE*(np.ones(n_x)+2*np.random.rand(n_x))
A_uncertain,B_uncertain,eig_A_uncertain = poles_to_linear_sys(poles_uncertain,Ts=TS)
theta0 = ca.DM(ca.vertcat(ca.vec(A_uncertain),ca.vec(B_uncertain)))

# another option is adding uncertainty to theta directly
# theta_uncertainty = THETA_UNCERTAINTY_RANGE*(np.ones(theta.shape)+2*np.random.rand(*theta.shape))
# theta0 = ca.DM( np.multiply(theta_uncertainty,np.array(true_theta)) )

print(f'Eigenvalues of uncertain system: {eig_A_uncertain}')
print(f'Initial condition: {x0}')
print(f'Initial parameter estimate: {theta0}')

# sample noise if requested
if NOISE:
    w0 = ca.horzsplit(NOISE_MAG*(2*np.random.rand(dyn.dim['w'],UPPER_HORIZON)-np.ones((dyn.dim['w'],UPPER_HORIZON))))

### CREATE MPC -----------------------------------------------------------------------------

# upper level cost
Q_true = 10*ca.DM.eye(n_x)
R_true = 1

# constraints are simple bounds on state and input
x_max = 5*ca.DM.ones(n_x,1)
x_min = -x_max
u_max = 1
u_min = -u_max

# parameter = terminal state cost and input cost
c_q = ca.SX.sym('c_q',int(n_x*(n_x+1)/2),1)
c_r = ca.SX.sym('c_r',1,1)

# stage cost (state)
Qx = [Q_true] * (MPC_HORIZON-1)

# stage cost (input)
Ru = c_r**2 + 1e-6

# create parameter
p = ca.vcat([c_q,c_r])
pf = theta

# MPC terminal cost
Qn = param2terminal_cost(c_q) + 0.01*ca.SX.eye(n_x)

# append to Qx
Qx.append(Qn)

# add to mpc dictionary
cost = {'Qx': Qx, 'Ru':Ru, 's_quad':MPC_S_QUAD, 's_lin':MPC_S_LIN}
# cost = {'Qx': Qx, 'Ru':Ru}

# turn bounds into polyhedral constraints
Hx,hx,Hu,hu = bound2poly(x_max,x_min,u_max,u_min)

# add to mpc dictionary
cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu, 'Hx_e':ca.SX.eye(hx.shape[0])}
# cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu}

# create QP ingredients
ing = Ingredients(horizon=MPC_HORIZON,dynamics=dyn,cost=cost,constraints=cst)

# create options
qp_options = {'compile_qp_sparse':COMPILE_QP_SPARSE,
              'compile_qp_dense':COMPILE_QP_DENSE,
              'compile_jac':COMPILE_JAC,
              'solver':SOLVER}

# create MPC
mpc = QP(ingredients=ing,p=p,pf=pf,options=qp_options)


### UPPER LEVEL -----------------------------------------------------------

# create upper level
upper_level = UpperLevel(p=p,pf=pf,horizon=UPPER_HORIZON,mpc=mpc)

# extract linearized dynamics at the origin
A = dyn.A_nom(ca.DM(n_x,1),ca.DM(n_u,1),theta0)
B = dyn.B_nom(ca.DM(n_x,1),ca.DM(n_u,1),theta0)

# verify that A and B match the matrices computed above
assert ca.mmax(ca.fabs(ca.DM(A-A_uncertain)))==0, 'Nominal matrix A is incorrect.'
assert ca.mmax(ca.fabs(ca.DM(B-B_uncertain)))==0, 'Nominal matrix B is incorrect.'

# compute terminal cost initialization
p_init = ca.vertcat(dare2param(A,B,Q_true,R_true),1e-1)#ca.vertcat(ca.DM.ones(p.shape[0]-1,1)*1e-3,1)#

# extract closed-loop variables for upper level
x_cl = ca.vec(upper_level.param['x_cl'])
u_cl = ca.vec(upper_level.param['u_cl'])

track_cost, cst_viol_l1, cst_viol_l2 = quad_cost_and_bounds(Q_true,R_true,x_cl,u_cl,x_max,x_min)

# put together
cost = track_cost + L2_PENALTY*cst_viol_l2 + L1_PENALTY*cst_viol_l1

# create upper-level constraints
Hx,hx,_,_ = bound2poly(x_max,x_min,u_max,u_min,UPPER_HORIZON+1)
_,_,Hu,hu = bound2poly(x_max,x_min,u_max,u_min,UPPER_HORIZON)
cst_viol = ca.vcat([Hx@ca.vec(x_cl)-hx,Hu@ca.vec(u_cl)-hu])

# store in upper-level
upper_level.set_cost(cost,track_cost,cst_viol)

# create algorithm
p = upper_level.param['p']
j_p = upper_level.param['J_p']
k = upper_level.param['k']

# create update functions
parameter_update_gd, parameter_init_gd = gradient_descent(rho=0.0001,eta=0.8,log=True)
parameter_update_robust, parameter_init_robust = robust_gradient_descent(rho=RHO,eta=0.51,n_models=N_MODELS,n_p=p.shape[0],log=True)

# create system identification
sys_id_update, sys_id_init, _ = rls(
    dynamics=dyn,
    horizon=UPPER_HORIZON,
    lam=0.1,
    theta0=theta0,
    jit=False,
    idx_pf=range(theta0.shape[0]))

# alternative system identification
sys_id_update_robust, sys_id_init_robust, _ = rls_robust(
    dynamics=dyn,
    n_models=N_MODELS,
    R=1,
    S=1,
    delta=0.01,
    horizon=UPPER_HORIZON,
    lam=3,
    theta0=theta0,
    jit=False,
    idx_pf=range(theta0.shape[0]))

# update upper-level algorithm
# upper_level.set_alg(
#     parameter_update=parameter_update_gd,
#     parameter_init=parameter_init_gd,
#     sys_id_update=sys_id_update,
#     sys_id_init=sys_id_init)

upper_level.set_alg(
    parameter_update=parameter_update_robust,
    parameter_init=parameter_init_robust,
    sys_id_update=sys_id_update_robust,
    sys_id_init=sys_id_init_robust)

# test derivatives
# out = tests.derivatives(mod)


### CREATE SCENARIO -----------------------------------------------------------

scenario = Scenario(dyn,mpc,upper_level)

# initialize
init_dict = {'p':p_init,'pf':theta0,'x': x0,'theta':theta0}
if NOISE:
    init_dict['w'] = w0
init_dict['theta'] = [init_dict['theta']] * N_MODELS # needed for compatibility
scenario.set_init(init_dict)

# update options
sim_options = {'use_true_model':False,'max_k':ITERATIONS,'true_theta':np.array(true_theta),'simulate_parallel_models':True}
scenario.update_options(sim_options)

# # run first simulation
# sim, out_dict, qp_failed = scenario.simulate()

# if qp_failed:
#     raise ValueError('QP failed')

# while True:

#     # get cost
#     cost,track_cost,cst_viol = scenario.upper_level.cost(sim)

#     # run a single update
#     sim.j_p = scenario._mapped['j_cost'](sim)


# test closed loop
sim_list,_,p_best = scenario.closed_loop()

# # modify upper-level algorithm
# upper_level.set_alg(
#     parameter_update=parameter_update_robust,
#     parameter_init=parameter_init_robust,
#     sys_id_update=sys_id_update_robust,
#     sys_id_init=sys_id_init_robust)

# # update scenario
# scenario.update(upper_level=upper_level)

# # re-apply initialization
# init_dict['theta'] = [init_dict['theta']] * N_MODELS # needed for compatibility
# scenario.set_init(init_dict)

# # test again
# simulation_options = {'simulate_parallel_models':True,'use_true_model':False,'max_k':ITERATIONS,'true_theta':np.array(true_theta)}
# sim_list,_,p_best = scenario.closed_loop(options=simulation_options)

# retrieve thetas
# estimation_error = [ca.norm_2(ca.fabs(elem.psi['theta']-true_theta)) for elem in sim_list]
# print(estimation_error)

# get last value of p
# p_final = SIM[-1].p