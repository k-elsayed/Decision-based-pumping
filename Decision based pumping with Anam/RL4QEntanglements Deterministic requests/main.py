import numpy as np
import time
import json

from utils import create_file_path, save_params_to_file
from simulating_envs.pois_env import Q_env
from solvers import rllib_agent_action_ppo
from simulating_envs.predicting_expon_requests import ExponPred

# def simulate_sequences_till_time_T(R1_arr_rate, R2_arr_rate, delta_t_g, T, p_g, sample_size):
#     # Use the exponential distribution to generate inter-arrival times
#     R1_simulated_iat_times = np.random.exponential(scale=1/R1_arr_rate, size=sample_size)
#     R2_simulated_iat_times = np.random.exponential(scale=1/R2_arr_rate, size=sample_size)
#     R1_event_times = np.cumsum(R1_simulated_iat_times)
#     R2_event_times = np.cumsum(R2_simulated_iat_times)
#     k_geom_samples = np.random.geometric(p_g, size=sample_size)
#     tau_g_occurances = k_geom_samples * delta_t_g
#     tau_g_time_series = np.cumsum(tau_g_occurances)
#
#     R1_event_times = R1_event_times[R1_event_times < T]
#     R2_event_times = R2_event_times[R2_event_times < T]
#     tau_g_time_series = tau_g_time_series[tau_g_time_series < T]
#
#     # Tag each element in the arrays with an identifier for its original array
#     tagged_array1 = [(num, 'R_1') for num in R1_event_times]
#     tagged_array2 = [(num, 'R_2') for num in R2_event_times]
#     tagged_array3 = [(num, 'TE') for num in tau_g_time_series]
#
#     # Merge the tagged arrays
#     merged_tagged_array = tagged_array1 + tagged_array2 + tagged_array3
#     merged_tagged_request_array = tagged_array1 + tagged_array2
#
#
#     # Sort the merged array by the numbers
#     sorted_tagged_array = sorted(merged_tagged_array, key=lambda x: x[0])
#     sorted_tagged_request_array = sorted(merged_tagged_request_array, key=lambda x: x[0])
#
#     # Split the sorted, tagged array into two arrays: numbers and tags
#     event_time_series = np.array([item[0] for item in sorted_tagged_array])
#     event_sequence = np.array([item[1] for item in sorted_tagged_array])
#
#     request_times = np.array([item[0] for item in sorted_tagged_request_array])
#     request_sequence = np.array([item[1] for item in sorted_tagged_request_array])
#
#     return event_time_series, event_sequence, request_times, request_sequence, tau_g_time_series, R1_simulated_iat_times, R2_simulated_iat_times

def save_data(results_dir, R1_arr_rate, R2_arr_rate, delta_t, delta_t_g, T,
              trained_policy, F_E_0, F_R_1, F_R_2, eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, c, sample_size):
    run_params = {
        'R1_arr_rate': R1_arr_rate,
        'R2_arr_rate': R2_arr_rate,
        'delta_t': delta_t,
        'delta_t_g': delta_t_g,
        'T': T,
        'trained_policy': trained_policy,
        'F_E_0': F_E_0,
        'F_R_1': F_R_1,
        'F_R_2': F_R_2,
        'eta_db':eta_db,
        'eta_linear':eta_linear,
        'l_km':l_km,
        'p_g':p_g,
        't_c': t_c,
        'c': c,
        'sample_size': sample_size,
        'ppo_actions_discrete': ppo_actions_discrete,
    }
    save_params_to_file(run_params, results_dir.joinpath('run_params.json'))  # use if you want to save datewise

def env_creator(env_config):
    return Q_env(sample_size, R1_arr_rate, R2_arr_rate, delta_t_g, T, F_E_0, F_R_1, F_R_2, pred_env,
                 eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete)

def create_env(sample_size, R1_arr_rate, R2_arr_rate, delta_t, T, F_E_0, F_R_1, F_R_2, pred_env,
               eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete):
    # create env
    env = Q_env(sample_size, R1_arr_rate, R2_arr_rate, delta_t, T, F_E_0, F_R_1, F_R_2, pred_env,
                eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete)
    return env

def create_pred_env(R1_arr_rate=0.4, R2_arr_rate=0.8, T=1.0):
    pred_env = ExponPred(R1_arr_rate, R2_arr_rate, T)
    return pred_env

def train_env(env_creator, results_dir):
    solver = rllib_agent_action_ppo.RLLibSolver(env_creator, results_dir=results_dir)
    begin = time.time()
    avg_reward, min_reward, max_reward, trainer = solver.solve(env)
    print(time.time() - begin)

if __name__ == '__main__':
    # c=2 * 10**8
    # l_km = 1
    # t_c = 1     # in ms
    # eta_db = 0.15
    # eta_linear = 10 ** (eta_db * (l_km / 10))
    # p_g = np.exp(-eta_linear)
    # delta_t = (l_km * 1000) / c   # time slot
    # delta_t_g = delta_t/p_g
    # R1_arr_rate = 4000  # poisson rate \lambda_1, events/ms
    # R2_arr_rate = 6000  # poisson rate \lambda_2, events/ms
    # # assert(R1_arr_rate + R2_arr_rate < 1 / delta_t_g)
    # F_E_0 = 0.75
    # F_R_1 = F_E_0 + 0.2
    # F_R_2 = F_E_0 - 0.2
    # sample_size = 1000
    # T = delta_t_g * sample_size  # time horizon in ms

    run_param_path = '/Users/anam/PycharmProjects/RL4QEntanglements/Results/Parameter_Files/TwoPoisson/run_params_TwoPois.json'
    run_param = json.load(open(run_param_path, 'r'))
    c = run_param['c']
    l_km = run_param['l_km']
    t_c = run_param['t_c']     # in ms
    eta_db = run_param['eta_db']
    eta_linear = run_param['eta_linear']
    p_g = run_param['p_g']
    delta_t = run_param['delta_t']  # time slot
    delta_t_g = run_param['delta_t_g']
    R1_arr_rate = run_param['R1_arr_rate']
    R2_arr_rate = run_param['R2_arr_rate']
    F_E_0 = run_param['F_E_0']
    F_R_1 = run_param['F_R_1']
    F_R_2 = run_param['F_R_2']
    sample_size = run_param['sample_size']
    T = run_param['T']

    trained_policy = 'QE_Pois'
    ppo_actions_discrete = False
    results_dir = create_file_path(trained_policy)

    save_data(results_dir, R1_arr_rate,
              R2_arr_rate, delta_t, delta_t_g, T, trained_policy,
              F_E_0, F_R_1, F_R_2, eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, c, sample_size)
    pred_env = create_pred_env(R1_arr_rate, R2_arr_rate, T)
    env = create_env(sample_size, R1_arr_rate,
                     R2_arr_rate, delta_t_g, T, F_E_0, F_R_1, F_R_2,
                     pred_env, eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete)
    train_env(env_creator, results_dir=results_dir)


