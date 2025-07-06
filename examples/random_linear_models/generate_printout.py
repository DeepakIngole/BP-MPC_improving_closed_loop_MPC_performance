import pickle

result_name = '/home/riccardoz/Documents/Git/BP-MPC_improving_closed_loop_MPC_performance/.results/gd_CE_2025_07_05_13_15_49_random_linear_models_NX_4_POLE_MAG_-5.0_to_1.0_N_MODELS_50_NOISE.pkl'

with open(result_name, 'rb') as f:
    result = pickle.load(f)

print(*result['printout'], sep='\n')