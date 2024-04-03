import os
import shutil
import time

import numpy as np
import ray
from ray.rllib.agents.ppo import ppo
# from ray.rllib.algorithms import ppo

from ray.tune import register_env


from utils import custom_log_creator
from testing_trainer.training_progress import training_progress_func

class RLLibSolver():
    """
    Approximate deterministic solutions using Rllib
    """
    def __init__(self, env_creator, **kwargs):
        super().__init__()
        self.env_creator = env_creator
        self.kwargs = kwargs
    def solve(self, env, **kwargs):
        ray.init(local_mode=False)
        # ray.init(local_mode=False, ignore_reinit_error=True)
        register_env("QE", self.env_creator)
        trainer = ppo.PPOTrainer(env="QE", logger_creator=custom_log_creator(self.kwargs.get('results_dir')),
                                 config={
                                     'framework': 'tf',
                                     "num_workers": 24,
                                     }
                                 )
        # config = PPOConfig()
        # config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3,
        #                          train_batch_size=128)
        # config = config.resources(num_gpus=0)
        # config = config.rollouts(num_rollout_workers=1)
        #
        #
        # trainer = config.build(env="QE")

        logs = []
        avg_reward = []
        min_reward = []
        max_reward = []
        min_rew = None
        min_rew_thr = 0.5 # in percentage
        best_results_dir = self.kwargs.get('results_dir').joinpath('best_result')
        best_results_dir.mkdir(exist_ok=False)
        begin = time.time()
        a = True
        i = 0
        best_result_iter = 0

        while a:
            print('i', i)
            i += 1
            log = trainer.train()
            time_elapsed = time.time() - begin
            print(f'step: {i}; time elapsed: {time_elapsed/60:.2f} mins')
            logs.append(log)

            if not np.isnan(log['episode_reward_mean']):
                if min_rew is None:
                    min_rew = log['episode_reward_mean']
                    trainer.save(checkpoint_dir=str(best_results_dir))
                elif min_rew - log['episode_reward_mean'] < min_rew_thr:
                    min_rew = log['episode_reward_mean']
                    shutil.rmtree(best_results_dir.joinpath(os.listdir(best_results_dir)[0]), ignore_errors=True)
                    trainer.save(checkpoint_dir=str(best_results_dir))
                    best_result_iter = trainer.get_state()['iteration']

            print('mean reward', log['episode_reward_mean'])
            avg_reward.append(log['episode_reward_mean'])
            min_reward.append(log['episode_reward_min'])
            max_reward.append(log['episode_reward_max'])

            if i % 20 == 0:
                training_pth = best_results_dir.parent
                training_progress_func(training_pth)

            if i % 49 == 0:
                checkpoint_path = trainer.save(checkpoint_dir=str(self.kwargs.get('results_dir')))

            if trainer.get_state()['iteration'] - best_result_iter > 200:
                break
        return avg_reward, min_reward, max_reward, trainer
