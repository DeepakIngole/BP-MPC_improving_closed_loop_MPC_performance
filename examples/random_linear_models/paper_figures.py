import pickle
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# load results
model_path = '/home/riccardoz/Documents/Git/BP-MPC_improving_closed_loop_MPC_performance/.models/2025_06_17_08_15_43_random_linear_models_NX_4_POLE_MAG_-5.0_to_1.0_N_MODELS_100.pkl'
results_ce_path = '/home/riccardoz/Documents/Git/BP-MPC_improving_closed_loop_MPC_performance/.results/gd_CE_2025_06_17_08_15_43_random_linear_models_NX_4_POLE_MAG_-5.0_to_1.0_N_MODELS_100.pkl'
results_no_ce_path = '/home/riccardoz/Documents/Git/BP-MPC_improving_closed_loop_MPC_performance/.results/gd_NO_CE_2025_06_17_08_15_43_random_linear_models_NX_4_POLE_MAG_-5.0_to_1.0_N_MODELS_100.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(results_ce_path, 'rb') as f:
    results_ce = pickle.load(f)['results']
with open(results_no_ce_path, 'rb') as f:
    results_no_ce = pickle.load(f)['results']

theta_true = [np.array(elem['theta_true']) for elem in model]
n_theta = theta_true[0].shape[0]
n_iter = len(results_ce[0]['cost'])

cost_ce = np.array([np.min(np.array(elem['cost'])) for elem in results_ce])
cost_ce_first = np.array([elem['cost'][0] for elem in results_ce]).squeeze()
optimal_cost_ce = np.array([elem['best_cost'] for elem in results_ce])

theta_difference_ce = np.dstack([np.hstack(elem['theta']-theta_true_single) for elem,theta_true_single in zip(results_ce,theta_true) if len(elem['cost']) == n_iter])
theta_error_ce = np.linalg.norm(theta_difference_ce, axis=0)

cost_no_ce = np.array([np.min(np.array(elem['cost'])) for elem in results_no_ce])
cost_no_ce_first = np.array([elem['cost'][0] for elem in results_no_ce]).squeeze()
optimal_cost_no_ce = np.array([elem['best_cost'] for elem in results_no_ce])

theta_difference_no_ce = np.dstack([np.hstack(elem['p'])[-n_theta:,:]-theta_true_single for elem,theta_true_single in zip(results_no_ce,theta_true) if len(elem['cost']) == n_iter])
theta_error_no_ce = np.linalg.norm(theta_difference_no_ce, axis=0)