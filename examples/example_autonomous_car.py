import sys
import os

# add root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scenario import Scenario
from src.dynamics import Dynamics
from src.qp import QP
from src.ingredients import Ingredients
from utils.cleanup import cleanup
from utils.cost_utils import quad_cost_and_bounds,bound2poly,param2terminal_cost,dare2param
# import tests.tests as tests
import dynamics.autonomous_car as autonomous_car
import casadi as ca
from src.plotter import Plotter
from src.upper_level import UpperLevel
import numpy as np
from utils.parameter_update import gradient_descent, adam
from utils.sys_id import rls

# cleanup jit files
cleanup()

# choose relative uncertainty on model
UNCERTAINTY_RANGE = 0.8

# choose constant velocity
VELOCITY = 10.0

# max iteration number
ITERATIONS = 50

# radius of uncertainty ball
BALL_RADIUS = 0.5

# choose initial GD stepsize
RHO = 0.001

# choose if sys id will be perfomed
SYS_ID = False
LAM = 5

# choose waypoints
WAYPOINTS = np.hsplit(np.array([[-4,-4],[0,-4],[4,-4],[4,0],[4,4],[0,4],[-4,4],[-4,2],[2,2],[2,-2],[0,-2],[-2,-2],[-2,0],[-4,0],[-4,-2],[-4,-4]]) / 4.0 * 15, 2)


### GENERATE MODEL ------------------------------------------------------------------------

# create dictionary with parameters of cart pendulum
# uncertainty = ca.DM(np.random.rand(8)*UNCERTAINTY_RANGE*2)-ca.DM.ones(8)*UNCERTAINTY_RANGE
uncertainty = ca.vertcat(0.74123508, -0.78708669, -0.97033045, -0.11980179, 0.75357767, -0.60515449, 0.67895028, 0.71907294)*UNCERTAINTY_RANGE
dyn_dict, true_theta, nominal_theta = autonomous_car.dynamics(uncertainty=uncertainty,velocity=VELOCITY)

print(ca.norm_2(true_theta-nominal_theta))

# create dynamics object
dyn = Dynamics(dyn_dict)

# get state and input dimensions
n_x, n_u, n_w, n_theta = dyn.dim['x'], dyn.dim['u'], dyn.dim['w'], dyn.dim['theta']

# run interpolation
waypoints_interpolated, r_s, tangent_direction = autonomous_car.generate_waypoints(waypoints=WAYPOINTS,velocity=VELOCITY)

# initial error is zero
x0 = ca.vertcat(0.75,0,0,0)

# disturbance is the path curvature
w0 = ca.vertsplit(ca.DM(r_s))

# upper-level horizon
upper_horizon = len(w0)


### CREATE MPC -----------------------------------------------------------------------------

# upper level cost
Q_true = ca.diag(ca.vertcat(1,1,1,1))
R_true = 1e-6

# mpc horizon
mpc_horizon = 5

# constraints are simple bounds on state and input
x_max = ca.vertcat(1,5,1,2.5)
x_min = -x_max
u_max = ca.pi / 5
u_min = -u_max

# parameter = terminal state cost and input cost
c_q = ca.SX.sym('c_q',n_x,1)#ca.SX.sym('c_q',int(n_x*(n_x+1)/2),1)
c_q_n = ca.SX.sym('c_q',int(n_x*(n_x+1)/2),1)
c_r = ca.SX.sym('c_r',1,1)

# stage cost (state)
# Qx = [Q_true] * (mpc_horizon-1)
Qx = [ca.diag(c_q)] * (mpc_horizon-1)

# stage cost (input)
Ru = c_r**2 + 1e-6

# create parameters
p = ca.vcat([c_q,c_q_n,c_r])
pf = dyn_dict['theta']

# MPC terminal cost
Qn = param2terminal_cost(c_q_n) + 0.01*ca.SX.eye(n_x)

# append to Qx
Qx.append(Qn)

# slack penalties
c_lin = 25
c_quad = 25

# add to mpc dictionary
cost = {'Qx': Qx, 'Ru':Ru}

# turn bounds into polyhedral constraints
Hx,hx,Hu,hu = bound2poly(x_max,x_min,u_max,u_min)

# add to mpc dictionary
cst = {'hx':hx, 'Hx':Hx, 'hu':hu, 'Hu':Hu}

# create QP ingredients
ing = Ingredients(horizon=mpc_horizon, dynamics=dyn, cost=cost, constraints=cst)

# create MPC
MPC = QP(ingredients=ing, p=p, pf=pf)


### UPPER LEVEL -----------------------------------------------------------

# create upper level
upper_level = UpperLevel(p=p, pf=pf, horizon=upper_horizon, mpc=MPC)

# extract linearized dynamics at the origin
A = dyn.A_nom(ca.DM(n_x,1),ca.DM(n_u,1),nominal_theta)
B = dyn.B_nom(ca.DM(n_x,1),ca.DM(n_u,1),nominal_theta)

# compute terminal cost initialization
p_init = ca.vertcat(ca.diag(Q_true),dare2param(A,B,Q_true,R_true),1e-3)
pf_init = nominal_theta

# extract closed-loop variables for upper level
x_cl = ca.vec(upper_level.param['x_cl'])
u_cl = ca.vec(upper_level.param['u_cl'])

track_cost, cst_viol_l1, cst_viol_l2 = quad_cost_and_bounds(Q_true,R_true,x_cl,u_cl,x_max,x_min)

# put together
cost = track_cost

# create upper-level constraints
Hx,hx,_,_ = bound2poly(x_max,x_min,u_max,u_min,upper_horizon+1)
_,_,Hu,hu = bound2poly(x_max,x_min,u_max,u_min,upper_horizon)
cst_viol = ca.vcat([Hx@ca.vec(x_cl)-hx,Hu@ca.vec(u_cl)-hu])

# store in upper-level
upper_level.set_cost(cost,track_cost,cst_viol)

# create algorithm
p = upper_level.param['p']
J_p = upper_level.param['J_p']
k = upper_level.param['k']

# choose update algorithm
parameter_update, parameter_init = gradient_descent(rho=RHO,eta=0.6,log=True)
# parameter_update, parameter_init = adam(alpha=0.15, beta_1=0.5, beta_2=0.8, epsilon=1e-6)

if SYS_ID:

    # create system identification algorithm
    sys_id_update, sys_id_init, _ = rls(
        dynamics=dyn,
        horizon=upper_horizon,
        lam=LAM,
        theta0=ca.DM(nominal_theta),
        jit=False,
        idx_pf=range(nominal_theta.shape[0])
    )

    upper_level.set_alg(
        parameter_update=parameter_update,
        parameter_init=parameter_init,
        sys_id_update=sys_id_update,
        sys_id_init=sys_id_init
    )

else:

    upper_level.set_alg(
        parameter_update=parameter_update,
        parameter_init=parameter_init
    )

### CREATE SCENARIO AND SIMULATE -----------------------------------------------------------

scenario = Scenario(dyn,MPC,upper_level)

# initialize
init_dict = {'p':p_init, 'pf':pf_init, 'x': x0, 'w': w0, 'theta':nominal_theta}
scenario.set_init(init_dict)

# simulate with initial parameter
sim_0,qp_data_sparse,_ = scenario.simulate()

Plotter.plot_car_trajectory(
    waypoints=waypoints_interpolated,
    tangent_direction=tangent_direction,
    sim=sim_0,
    path_constraint=1,
    show=False,
    options={'legend':'Untrained','color':'orange','linestyle':'-.'}
)

# create plot but do not show
# Plotter.plot_trajectory(sim_0,options={'x':[0,1,2,3],'x_legend':['Position untrained','Velocity untrained','Angle untrained','Angular velocity untrained'],'u':[0],'u_legend':['Force untrained'],'color':'blue'},show=False)

# simulation options
sim_options = {'save_memory': True, 'use_true_model': False, 'max_k': ITERATIONS, 'true_theta': np.array(ca.DM(true_theta))}

# test closed loop
sim_list,_,p_best,_ = scenario.closed_loop(options=sim_options)

# get last value of p
p_final = sim_list[-1].p

# create plots
# Plotter.plot_trajectory(sim_list[-1],options={'x':[0,1,2,3],'x_legend':['Position tuned','Velocity tuned','Angle tuned','Angular velocity tuned'],'u':[0],'u_legend':['Force tuned'],'color':'red'},show=False)
Plotter.plot_car_trajectory(
    waypoints=waypoints_interpolated,
    tangent_direction=tangent_direction,
    sim=sim_list[-1],
    path_constraint=1,
    show=True,
    options={'legend':'Trained','color':'red','linestyle':'--'}
)

raise Exception

# create nonlinear solver
NLP = scenario.make_trajectory_opt(w=w0)

# create warm start trajectories
x_warm = sim_list[-1].x
u_warm = sim_list[-1].u

# solve
nlp_out,nlp_solved = NLP(x0,x_warm,u_warm)
print('NLP solved correctly') if nlp_solved else print('NLP failed')

# plot best solution
Plotter.plot_trajectory(nlp_out,options={'x':[0,1,2,3],'x_legend':['Position best','Velocity best','Angle best','Angular velocity best'],'u':[0],'u_legend':['Force best'],'color':'orange'},show=True)