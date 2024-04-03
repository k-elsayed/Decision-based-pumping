from utils import save_params_to_file
import numpy as np

from utils import save_to_file, get_short_datetime


class ASPP:     #always serve, always pump
    def __init__(self, env_creator, prob_pump, **kwargs):
        super().__init__()
        self.prob_pump = prob_pump
        self.env_creator = env_creator
        self.kwargs = kwargs
        self.results_dir = self.kwargs.get('results_dir')

    def solve(self, env):
        dt = get_short_datetime()
        output_dir = self.results_dir.joinpath(dt)
        output_dir.mkdir(exist_ok=True)
        episode_timesteps = 100
        cum_reward_per_ep = np.zeros(episode_timesteps)
        mc = 100
        for j in range(mc):
            print(j)
            all_rewards = []
            curr_obs = env.reset()
            for i in range(episode_timesteps):
                action_all_agents = [[0,1], [1-self.prob_pump,self.prob_pump]]      #always serve, always pump
                curr_obs, joint_reward, _, _ = env.step(action_all_agents)
                all_rewards.append(joint_reward)  # per agent reward

            cum_reward_per_ep += np.cumsum(all_rewards)
            curr_output_json = {
                                "reward": all_rewards
                                }
            save_to_file(curr_output_json, output_dir, j)
        cum_reward_per_ep = cum_reward_per_ep/mc
        return cum_reward_per_ep



