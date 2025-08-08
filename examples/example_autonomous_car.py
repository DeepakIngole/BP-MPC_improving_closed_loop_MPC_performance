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
from utils.parameter_update import gradient_descent, minibatch_descent

# cleanup jit files
cleanup()

# choose relative uncertainty on model
UNCERTAINTY_RANGE = 0

# create dictionary with parameters of cart pendulum
dyn_dict, true_theta, nominal_theta = autonomous_car.dynamics(uncertainty=ca.DM(np.random.rand(8)*UNCERTAINTY_RANGE*2)-ca.DM.ones(8)*UNCERTAINTY_RANGE)

# create dynamics object
dyn = Dynamics(dyn_dict)

# get state and input dimensions
n_x, n_u, n_w, n_theta = dyn.dim['x'], dyn.dim['u'], dyn.dim['w'], dyn.dim['theta']

# upper-level horizon
upper_horizon = 170

# set initial conditions
x0 = ca.DM(n_x,1)
w0 = ca.horzsplit(ca.DM(n_w,upper_horizon))


### CREATE MPC -----------------------------------------------------------------------------

# upper level cost
Q_true = ca.diag(ca.vertcat(10,1,10,1))
R_true = 1e-6

# mpc horizon
mpc_horizon = 20

# constraints are simple bounds on state and input
x_max = ca.vertcat(4,20,1,1)
x_min = -x_max
u_max = ca.pi / 4
u_min = -u_max

# parameter = terminal state cost and input cost
c_q = ca.SX.sym('c_q',int(n_x*(n_x+1)/2),1)
c_r = ca.SX.sym('c_r',1,1)

# stage cost (state)
Qx = [Q_true] * (mpc_horizon-1)

# stage cost (input)
Ru = c_r**2 + 1e-6

# create parameters
p = ca.vcat([c_q,c_r])
pf = dyn_dict['theta']

# MPC terminal cost
Qn = param2terminal_cost(c_q) + 0.01*ca.SX.eye(n_x)

# append to Qx
Qx.append(Qn)

# slack penalties
c_lin = 15
c_quad = 5

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
A = dyn.A_nom(ca.DM(n_x,1),ca.DM(n_u,1))
B = dyn.B_nom(ca.DM(n_x,1),ca.DM(n_u,1))

# compute terminal cost initialization
p_init = ca.vertcat(dare2param(A,B,Q_true,R_true),1e-3)

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

# create update function
upper_level.set_alg(*gradient_descent(rho=0.0001,eta=0.51,log=True))
# upper_level.set_alg(*minibatch_descent(rho=0.0001,eta=0.51,log=True,batch_size=2))

# test derivatives
# # out = tests.derivatives(mod)


### CREATE SCENARIO -----------------------------------------------------------

scenario = Scenario(dyn,MPC,upper_level)

# initialize
init_dict = {'p':p_init,'x': x0, 'w': w0}
scenario.set_init(init_dict)

# simulate with initial parameter
S,qp_data_sparse,_ = scenario.simulate()

# create plot but do not show
Plotter.plotTrajectory(S,options={'x':[0,1,2,3],'x_legend':['Position untrained','Velocity untrained','Angle untrained','Angular velocity untrained'],'u':[0],'u_legend':['Force untrained'],'color':'blue'},show=False)