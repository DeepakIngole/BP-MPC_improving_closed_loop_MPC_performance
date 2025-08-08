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