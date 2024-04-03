import json
import os
import pickle as pickle
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune import register_env

import utils
from simulating_envs.pois_env import Q_env
from utils import save_to_file
from main_ppo import create_pred_env, create_env


class EvaluatePolicy:
    def __init__(self, trainer, results_dir, **run_parameters):
        """

        :param trainer: preloaded trainer
        :param results_dir: a directory with parameters and checkpoints
        :param run_parameters: preloaded run parameters for the environment
        """
        self.params = run_parameters
        self.trained_policy = self.params.pop('trained_policy', 'MF')
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.trainer = trainer
        self.equal_time = True


    @classmethod
    def from_checkpoint(cls, checkpoint_path):

        chk = checkpoint_path.split('checkpoint_')[1]
        checkpoint_path = Path(checkpoint_path)

        results_dir_suffix = 'Evaluation_checkpoint'+chk



        run_params_path = checkpoint_path.parent.parent.joinpath('run_params.json')
        with run_params_path.open('r') as jf:
            run_params = json.load(jf)

        trained_policy = run_params['trained_policy']
        delta_t = run_params.pop('delta_t', 0)
        c = run_params.pop('c', 0)
        # sample_size = run_params.pop('sample_size', 1000)
        pred_env = create_pred_env(run_params['request_arrival_rates'], run_params['T'])
        run_params['pred_env'] = pred_env

        with checkpoint_path.parent.parent.joinpath('params.pkl').open('rb') as pf:
            config = pickle.load(pf)

        _chkpnt_file = str(checkpoint_path.joinpath(checkpoint_path.stem.replace('_', '-')))
        config['env_config'] = run_params
        config['num_workers'] = 1

        ray.init(local_mode=False)
        if trained_policy in ('QE_Pois', 'QE_Pois_PPO'):
            register_env("EVAL", lambda x: cls.create_env(x))
            agent = ppo.PPOTrainer(config, env='EVAL')
        else:
            raise NotImplementedError

        agent.restore(_chkpnt_file)
        # run_params['trained_policy'] = trained_policy
        run_params['delta_t'] = delta_t
        run_params['c'] = c
        # run_params['sample_size'] = sample_size
        return cls(agent, checkpoint_path.parent.joinpath(results_dir_suffix), **run_params)

    @staticmethod
    def create_env(params):
        env = Q_env(**params)
        return env

    def single_run_test(self, env):
        all_rewards = []
        all_states = []
        all_policies = []
        all_normalised_policies = []
        state_policy_reward ={}
        obs = env.reset()
        flags = [0, 1]
        waiting_times = [0.1, 0.2]
        F_r_all = [0.4, 0.5, 0.6]
        F_e_all = [0.5, 0.6]
        if self.trained_policy == 'QE_Pois_PPO':
            i = 0
            for flag in flags:
                for waiting_time in waiting_times:
                    for F_r in F_r_all:
                        for F_e in F_e_all:
                            state_policy_reward['{}'.format(i)] = {}
                            state = np.array([waiting_time, F_r, flag, F_e], dtype=np.float32)
                            state_policy_reward['{}'.format(i)]['state'] = state.tolist()
                            policy = self.trainer.compute_single_action(state)
                            state_policy_reward['{}'.format(i)]['policy'] = [arr.tolist() for arr in policy]
                            normalised_policy = env.normalise_actions(policy)
                            state_policy_reward['{}'.format(i)]['normalised_policy'] = [arr.tolist() for arr in normalised_policy]
                            all_policies.append(policy)
                            all_normalised_policies.append(normalised_policy)
                            all_states.append(state)
                            _, reward, _, _ = env.step(policy)
                            all_rewards.append(reward)
                            state_policy_reward['{}'.format(i)]['reward'] = reward
                            i += 1

        else:
            raise NotImplementedError(f'Policy is not implemented for {self.params.get("trained_policy")}')

        return all_rewards, state_policy_reward

    def run_test(self, request_arrival_rates, delta_t_g, T, F_E_0, F_R_all,
                         pred_env, eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, sample_size, trained_policy):
        no_mc = 1
        test_env = create_env(sample_size, request_arrival_rates, delta_t_g, T, F_E_0, F_R_all,
                         pred_env, eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, trained_policy, eval=True)

        # number_of_events = 10000
        # sum_ep_rewards = np.zeros(number_of_events)
        # cum_reward_per_ep = np.zeros(number_of_events)
        sum_ep_rewards = None
        cum_reward_per_ep = None
        # output directory
        dt = utils.get_short_datetime()
        file_path = 'Evaluation_per_obs_{}'.format(dt)
        agent_results_dir = self.results_dir.joinpath(file_path)
        agent_results_dir.mkdir(parents=True, exist_ok=True)
        for i in range(no_mc):
            print(i)
            ep_rewards, state_policy_reward = self.single_run_test(test_env)

            # curr_output_json = {
            #                     "state_policy_reward": state_policy_reward
            #                     }
            # # saving to file
            # save_to_file(curr_output_json, agent_results_dir, i)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('curr_dir', type=str)
    args = parser.parse_args()

    current_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.curr_dir

    ep = EvaluatePolicy.from_checkpoint(output_dir)
    c = ep.params['c']
    l_km = ep.params['l_km']
    t_c = ep.params['t_c']     # in ms
    eta_db = ep.params['eta_db']
    eta_linear = ep.params['eta_linear']
    p_g = ep.params['p_g']
    delta_t = ep.params['delta_t']   # time slot
    delta_t_g = ep.params['delta_t_g']
    # R1_arr_rate = ep.params['R1_arr_rate']  # poisson rate \lambda_1, events/ms
    # R2_arr_rate = ep.params['R2_arr_rate']  # poisson rate \lambda_2, events/ms
    # assert(R1_arr_rate + R2_arr_rate < 1 / delta_t_g)
    F_E_0 = ep.params['F_E_0']
    # F_R_1 = ep.params['F_R_1']
    # F_R_2 = ep.params['F_R_2']
    sample_size = ep.params['sample_size']
    T = ep.params['T'] # time horizon in ms
    trained_policy = ep.trained_policy
    ppo_actions_discrete = ep.params['ppo_actions_discrete']
    pred_env = ep.params['pred_env']
    F_R_all = ep.params['F_R_all']
    request_arrival_rates = ep.params['request_arrival_rates']

    ep.run_test(request_arrival_rates, delta_t_g, T, F_E_0, F_R_all,
                         pred_env, eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, sample_size, trained_policy)
