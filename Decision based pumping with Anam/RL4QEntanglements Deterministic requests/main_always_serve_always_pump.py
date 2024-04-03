import json

from utils import create_file_path, save_params_to_file
from simulating_envs.pois_env import Q_env
from solvers import always_serve_always_pump
from simulating_envs.predicting_expon_requests import ExponPred


def save_data(results_dir, request_arrival_rates, delta_t, delta_t_g, T,
              trained_policy, F_E_0, F_R_all, eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, c, sample_size):
    run_params = {
        'request_arrival_rates': request_arrival_rates,
#        'R1_arr_rate': R1_arr_rate,
 #       'R2_arr_rate': R2_arr_rate,
        'delta_t': delta_t,
        'delta_t_g': delta_t_g,
        'T': T,
        'trained_policy': trained_policy,
        'F_E_0': F_E_0,
        'F_R_all':F_R_all,
        #'F_R_1': F_R_1,
        #'F_R_2': F_R_2,
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
    return Q_env(sample_size, request_arrival_rates, delta_t_g, T, F_E_0, F_R_all, pred_env,
                 eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, trained_policy)

def create_env(sample_size, request_arrival_rates, delta_t, T, F_E_0, F_R_all, pred_env,
               eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, trained_policy):
    # create env
    env = Q_env(sample_size, request_arrival_rates, delta_t, T, F_E_0, F_R_all, pred_env,
                eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, trained_policy)
    return env

def create_pred_env(request_arrival_rates, T=1.0):
#def create_pred_env(R1_arr_rate=0.4, R2_arr_rate=0.8, T=1.0):
    pred_env = ExponPred(request_arrival_rates, T)
    #pred_env = ExponPred(R1_arr_rate, R2_arr_rate, T)
    return pred_env

def train_env(env_creator, results_dir):
    solver = always_serve_always_pump.ASAP(env_creator, results_dir=results_dir)
    avg_reward = solver.solve(env)

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

    run_param_path = 'Results/Parameter_Files/TwoPoisson/run_params_4Pois.json'
    run_param = json.load(open(run_param_path, 'r'))
    c = run_param['c']
    l_km = run_param['l_km']
    t_c = run_param['t_c']  # in ms
    eta_db = run_param['eta_db']
    eta_linear = run_param['eta_linear']
    p_g = run_param['p_g']
    delta_t = run_param['delta_t']  # time slot
    delta_t_g = run_param['delta_t_g']
    #R1_arr_rate = run_param['R1_arr_rate']
    #R2_arr_rate = run_param['R2_arr_rate']
    F_E_0 = run_param['F_E_0']
    #F_R_1 = run_param['F_R_1']
    #F_R_2 = run_param['F_R_2']
    sample_size = run_param['sample_size']
    T = run_param['T']
    F_R_all = run_param['F_R_all']
    request_arrival_rates = run_param['request_arrival_rates']

    trained_policy = 'QE_Pois_ASAP'
    ppo_actions_discrete = False
    results_dir = create_file_path(trained_policy)

    save_data(results_dir, request_arrival_rates,
              #R1_arr_rate, R2_arr_rate,
              delta_t, delta_t_g, T, trained_policy,
              F_E_0,
              #F_R_1, F_R_2,
              F_R_all,
              eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, c, sample_size)
    pred_env = create_pred_env(request_arrival_rates, T)
#    pred_env = create_pred_env(R1_arr_rate, R2_arr_rate, T)
    env = create_env(sample_size,
                     request_arrival_rates,
                     #R1_arr_rate, R2_arr_rate,
                     delta_t_g, T, F_E_0,
                     #F_R_1, F_R_2,
                     F_R_all,
                     pred_env, eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, trained_policy)
    train_env(env_creator, results_dir=results_dir)


