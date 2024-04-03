"""This file contains all the functions required to retrieve stored data in json format and average them"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import utils


class Compile:
    def __init__(self, path, expected_iter=100):

        if type(path) is list:
            _files = []
            for i in range(len(path)):
                _path = Path(path[i])
                _filepaths = _path.glob('**/*.json')
                _files.extend([f for f in _filepaths if not (str(f.stem).startswith('._') or 'average' in str(f)
                                                             or 'eps' in str(f) or 'comp_only_one_data' in str(f))])
            self._filepaths = _files
        else:
            self.path = Path(path)
            _filepaths = self.path.glob('*.json')
            self._filepaths = [f for f in _filepaths if not (str(f.stem).startswith('._') or 'average' in str(f)
                                                             or 'eps' in str(f))]
        self.expected_iter = expected_iter
        print("Total files found: {}, Files expected: {}".format(len(self._filepaths), self.expected_iter))
        self.tot_files = len(self._filepaths)

    @staticmethod
    def load(path: Path):
        # print(path)
        with path.open("r") as jf:
            return json.load(jf)

    def get_time_steps(self, delta_t):
        max_time = 500
        time_steps = np.round(max_time/delta_t).astype(int)
        return time_steps

    def compile(self):
        # mc = 30
        mc = len(self._filepaths)
        accum_data = []
        individual_mc_cum_data = []
        all_cum_data = np.zeros(100)
        for path in tqdm(self._filepaths[:mc], desc='Compiling'):
            # print(path)
            if path.stem[0] == 'd':
                _data = self.load(path)
                cum_data = np.cumsum(_data['reward'])
                # cum_data = np.cumsum(_data['reward'][0:no_events])    # for close NM
                accum_data.extend(_data['reward'])
                # accum_data.extend(_data['reward'][0:no_events])       # for close NM
                individual_mc_cum_data.append(cum_data[-1])
                all_cum_data += cum_data
        all_cum_data = all_cum_data / self.tot_files
        # get std
        std_cumsum = np.std(individual_mc_cum_data)
        # get confidence interval
        ci = (std_cumsum / np.sqrt(len(individual_mc_cum_data))) * 2
        #get sum
        sum_cumsum = np.sum(individual_mc_cum_data, axis=0)
        # get avg
        avg_cumsum = sum_cumsum / len(individual_mc_cum_data)
        return std_cumsum, avg_cumsum, ci, all_cum_data
        # return self, accum_data




if __name__ == '__main__':

    utils.figure_configuration_ieee_standard()

    ### to plot the final cumulative reward for all policies
    colors = ['#2ca02c','#1f77b4','#ff7f0e']
    line_styles = ['dotted','solid', 'dashed']
    output_dir = '/home/atahir/RL4QEntanglements/Results/Plots'
    final_cum_reward_per_nm = []
    final_cum_reward_per_nm_positive = []
    std_cum_reward_per_nm = []
    ci_cum_reward_per_nm = []
    # paths in order: 'QE_Pois_PPO', 'QE_Pois_ASPP', 'QE_Pois_ASAP', 'QE_Pois_ASNP'
    all_dir = [
        '/home/atahir/RL4QEntanglements/Results/QE_Pois_ASNP/0318_180440/0318_180440/',
        '/home/atahir/RL4QEntanglements/Results/QE_Pois_ASAP/0318_101215/0318_101215/'
               ]
    algos = ['QE_Pois_ASNP', 'QE_Pois_ASAP']
    i = 0
    for curr_dir in all_dir:
        algo = algos[0]
        i += 1
        final_cum_reward_per_algo = []
        final_cum_reward_per_algo_positive = []
        std_cum_reward_per_algo = []
        ci_cum_reward_per_algo = []
        avg_compiler = Compile(curr_dir, expected_iter=100)
        std_cumsum, cum_rew_last, ci, all_cum_data = avg_compiler.compile()
        final_cum_reward_per_algo.append(cum_rew_last)
        final_cum_reward_per_algo_positive.append(cum_rew_last * -1)
        std_cum_reward_per_algo.append(std_cumsum)
        ci_cum_reward_per_algo.append(ci)
        final_cum_reward_per_nm.append(final_cum_reward_per_algo)
        final_cum_reward_per_nm_positive.append(final_cum_reward_per_algo_positive)
        std_cum_reward_per_nm.append(std_cum_reward_per_algo)
        ci_cum_reward_per_nm.append(ci_cum_reward_per_algo)

    new_labels = []
    for l in range(len(algos)):
        if algos[l] == 'QE_Pois_ASNP':
            label = 'ASNP'
        elif algos[l] == 'QE_Pois_ASAP':
            label = 'ASAP'
        elif algos[l] == 'QE_Pois_PPO':
            label = 'PPO'
        elif algos[l] == 'QE_Pois_ASPP':
            label = 'ASPP'
        else:
            label = algos[l]
        new_labels.append(label)
        plt.errorbar(l, final_cum_reward_per_nm_positive[l], ci_cum_reward_per_nm[l],
                     capsize=5, label=label, color=colors[l], linestyle=line_styles[l])

    plt.xticks(ticks=np.arange(len(algos)), labels=new_labels)
    plt.ylabel('Reward')
    plt.xlabel('Event number')
    plt.savefig(output_dir + '/reward_comparison.pdf')
    plt.close()


