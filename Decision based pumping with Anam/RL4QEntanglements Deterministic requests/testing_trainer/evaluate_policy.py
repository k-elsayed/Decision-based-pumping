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

    def single_run_test(self, env, testing_time_step):
        all_rewards = []
        obs = env.reset()
        for i in range(testing_time_step):
            if self.trained_policy == 'QE_Pois_PPO':
                print(obs)
                action_trainer = self.trainer.compute_single_action(obs)  # for mf
                print(action_trainer)
            else:
                raise NotImplementedError(f'Policy is not implemented for {self.params.get("trained_policy")}')

            obs, joint_reward_np, _, _ = env.step(action_trainer)
            all_rewards.append(joint_reward_np)

        cum_reward = np.cumsum(all_rewards)
        return cum_reward, all_rewards

    def run_test(self, request_arrival_rates, delta_t_g, T, F_E_0, F_R_all,
                         pred_env, eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, sample_size, trained_policy):
        no_mc = 1
        test_env = create_env(sample_size, request_arrival_rates, delta_t_g, T, F_E_0, F_R_all,
                         pred_env, eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, trained_policy, eval=True)

        number_of_events = sample_size
        sum_ep_rewards = np.zeros(number_of_events)
        cum_reward_per_ep = np.zeros(number_of_events)

        # output directory
        dt = utils.get_short_datetime()
        file_path = 'Evaluation_{}'.format(dt)
        agent_results_dir = self.results_dir.joinpath(file_path)
        agent_results_dir.mkdir(parents=True, exist_ok=True)
        for i in range(no_mc):
            print(i)
            cum_reward, ep_rewards = self.single_run_test(test_env, number_of_events)
            sum_ep_rewards += ep_rewards
            cum_reward_per_ep += cum_reward
            curr_output_json = {
                                "reward": ep_rewards
                                }
            # saving to file
            save_to_file(curr_output_json, agent_results_dir, i)

        # Take average of cum_reward over number of mc sims
        avg_cum_ep_reward = cum_reward_per_ep / no_mc


        plt.plot(np.arange(len(avg_cum_ep_reward)), avg_cum_ep_reward, 'g', label='Avg Cumulative reward')
        plt.legend()
        plt.savefig(agent_results_dir.joinpath('agents_avg_cum_reward_testing.pdf'))
        plt.close()


if __name__ == '__main__':

    parser = ArgumentParser()
    # parser.add_argument('c', type=int)
    # parser.add_argument('l_km', type=int)
    # parser.add_argument('t_c', type=int)
    # parser.add_argument('eta_db', type=float)
    # parser.add_argument('eta_linear', type=float)
    # parser.add_argument('p_g', type=float)
    # parser.add_argument('delta_t', type=float)
    # parser.add_argument('delta_t_g', type=float)
    # parser.add_argument('R1_arr_rate', type=float)
    # parser.add_argument('R2_arr_rate', type=float)
    # parser.add_argument('F_E_0', type=float)
    # parser.add_argument('F_R_1', type=float)
    # parser.add_argument('F_R_2', type=float)
    # parser.add_argument('T', type=float)
    # parser.add_argument('trained_policy', type=str)
    # parser.add_argument('sample_size', type=int)
    # parser.add_argument('ppo_actions_discrete', type=str)
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
