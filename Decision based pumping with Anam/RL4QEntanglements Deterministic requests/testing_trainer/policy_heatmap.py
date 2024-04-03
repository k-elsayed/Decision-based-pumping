import json
import os
import pickle as pickle
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
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
        run_params['delta_t'] = delta_t
        run_params['c'] = c
        return cls(agent, checkpoint_path.parent.joinpath(results_dir_suffix), **run_params)

    @staticmethod
    def create_env(params):
        env = Q_env(**params)
        return env

    def single_run_test(self, env, k):
        obs = env.reset()
        flag = 1
        waiting_time = 0.1
        F_r_all = [0.1,0.2,0.3,0.4, 0.5, 0.6,0.7,0.8,0.9]
        F_e_all = [0.1,0.2,0.3,0.4, 0.5, 0.6,0.7,0.8,0.9]
        policy_serve = []
        policy_pump = []

        if self.trained_policy == 'QE_Pois_PPO':
            i = 0
            for F_r in F_r_all:
                for F_e in F_e_all:
                    state = np.array([waiting_time, F_r, flag, F_e], dtype=np.float32)
                    policy = self.trainer.compute_single_action(state)
                    normalised_policy = env.normalise_actions(policy)
                    policy_pump.append(normalised_policy[1][1])
                    policy_serve.append(normalised_policy[0][1])
                    _, reward, _, _ = env.step(policy)
                    i += 1

        else:
            raise NotImplementedError(f'Policy is not implemented for {self.params.get("trained_policy")}')

        # x_tick_labels = [f'{x}' for x in np.array(F_r_all)]  # For columns
        # y_tick_labels = [f'{y}' for y in np.array(F_e_all)]
        # heatmap_data = np.array(policy_pump).reshape([9, 9])
        # plt.figure(figsize=(10, 8))  # Size of the figure
        # sns.heatmap(heatmap_data, cmap='viridis', annot=True,  xticklabels=x_tick_labels, yticklabels=y_tick_labels)
        # plt.xlabel("Required fidelity, $F_r$")
        # plt.ylabel("Available fidelity, $F_e$")
        # plt.title("Waiting time: {}".format(waiting_time))
        # plt.savefig(self.results_dir.parent.joinpath("heatmap_pump{}.pdf".format(k)))
        #
        # # heatmap_data = np.array(policy_serve).reshape([9, 9])
        # # plt.figure(figsize=(10, 8))  # Size of the figure
        # # sns.heatmap(heatmap_data, cmap='viridis', annot=True,  xticklabels=x_tick_labels, yticklabels=y_tick_labels)
        # # plt.xlabel("Required fidelity, $F_r$")
        # # plt.ylabel("Available fidelity, $F_e$")
        # # plt.title("Waiting time: {}".format(waiting_time))
        # # plt.savefig(self.results_dir.parent.joinpath("heatmap_serve{}.pdf".format(k)))
        #


        # # Assuming F_e_all, F_r_all, and policy_pump are defined as per your snippet
        F_e_all = np.array(F_e_all)  # Your existing y-axis values
        F_r_all = np.array(F_r_all)  # Your existing x-axis values
        # pumping heatmap
        heatmap_data = np.array(policy_pump).reshape([9, 9])  # Your heatmap data
        # Create a DataFrame for easy reordering
        df = pd.DataFrame(heatmap_data, index=F_e_all, columns=F_r_all)
        # Decide on the new order (here, sorted in descending order for illustration)
        sorted_index = np.argsort(F_e_all)[::-1]  # This sorts F_e_all in descending order
        new_order = F_e_all[sorted_index]
        # Reorder the DataFrame
        df_reordered = df.loc[new_order]
        # Now, update your x_tick_labels and y_tick_labels based on the new order
        x_tick_labels = [f'{x}' for x in df_reordered.columns]  # Columns remain unchanged
        y_tick_labels = [f'{y}' for y in df_reordered.index]  # Rows now reflect the new order
        # Convert the reordered DataFrame back to a NumPy array for seaborn
        heatmap_data_reordered = df_reordered.values
        # Plotting the reordered heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data_reordered, cmap='viridis', annot=True, xticklabels=x_tick_labels,
                    yticklabels=y_tick_labels)
        plt.xlabel("Required fidelity, $F_r$")
        plt.ylabel("Available fidelity, $F_e$")
        plt.title(f"Waiting time: {waiting_time}")
        plt.savefig(self.results_dir.parent.joinpath("heatmap_pump{}.pdf".format(k)))

        # serving heatmap
        heatmap_data = np.array(policy_serve).reshape([9, 9])  # Your heatmap data
        # Create a DataFrame for easy reordering
        df = pd.DataFrame(heatmap_data, index=F_e_all, columns=F_r_all)
        # Decide on the new order (here, sorted in descending order for illustration)
        sorted_index = np.argsort(F_e_all)[::-1]  # This sorts F_e_all in descending order
        new_order = F_e_all[sorted_index]
        # Reorder the DataFrame
        df_reordered = df.loc[new_order]
        # Now, update your x_tick_labels and y_tick_labels based on the new order
        x_tick_labels = [f'{x}' for x in df_reordered.columns]  # Columns remain unchanged
        y_tick_labels = [f'{y}' for y in df_reordered.index]  # Rows now reflect the new order
        # Convert the reordered DataFrame back to a NumPy array for seaborn
        heatmap_data_reordered = df_reordered.values
        # Plotting the reordered heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data_reordered, cmap='viridis', annot=True, xticklabels=x_tick_labels,
                    yticklabels=y_tick_labels)
        plt.xlabel("Required fidelity, $F_r$")
        plt.ylabel("Available fidelity, $F_e$")
        plt.title(f"Waiting time: {waiting_time}")
        plt.savefig(self.results_dir.parent.joinpath("heatmap_serve{}.pdf".format(k)))

    def run_test(self, request_arrival_rates, delta_t_g, T, F_E_0, F_R_all,
                         pred_env, eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, sample_size, trained_policy):
        no_mc = 1
        test_env = create_env(sample_size, request_arrival_rates, delta_t_g, T, F_E_0, F_R_all,
                         pred_env, eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, trained_policy, eval=True)

        dt = utils.get_short_datetime()
        file_path = 'Evaluation_per_obs_{}'.format(dt)
        agent_results_dir = self.results_dir.joinpath(file_path)
        agent_results_dir.mkdir(parents=True, exist_ok=True)
        for i in range(no_mc):
            print(i)
            self.single_run_test(test_env, i)

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
    F_E_0 = ep.params['F_E_0']
    sample_size = ep.params['sample_size']
    T = ep.params['T'] # time horizon in ms
    trained_policy = ep.trained_policy
    ppo_actions_discrete = ep.params['ppo_actions_discrete']
    pred_env = ep.params['pred_env']
    F_R_all = ep.params['F_R_all']
    request_arrival_rates = ep.params['request_arrival_rates']

    ep.run_test(request_arrival_rates, delta_t_g, T, F_E_0, F_R_all,
                         pred_env, eta_db, eta_linear, l_km, p_g, t_c, ppo_actions_discrete, sample_size, trained_policy)
