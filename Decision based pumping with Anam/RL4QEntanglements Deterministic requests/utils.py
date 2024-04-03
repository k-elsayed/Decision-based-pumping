""" A place to store all utility functions """
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from ray.tune.logger import UnifiedLogger
import matplotlib

def create_directory(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def get_datetime():
    """
    Returns current data and time as e.g.: '2019-4-17_21_40_56'
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_short_datetime():
    """
    Returns current data and time as e.g.: '0417_214056'
    """
    return datetime.now().strftime("%m%d_%H%M%S")


def ndarray_to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, np.int_):
        return int(x)
    # elif isinstance(x, Iterable) and any(isinstance(y, np.ndarray) for y in x):
    #     return [y.tolist() if isinstance(y, np.ndarray) else y for y in x]
    return x


def recursive_conversion(d, func=ndarray_to_list):
    if isinstance(d, dict):
        return {k: recursive_conversion(v) for k, v in d.items()}
    return func(d)


def create_file_path(config):
    script_path = Path(__file__).absolute().parent
    results_dir = script_path.joinpath('Results')
    results_dir.mkdir(exist_ok=True)
    run_time = get_short_datetime()
    file_name = '{config}'.format(config=config)
    output_dir = results_dir.joinpath(file_name)
    create_directory(output_dir)
    output_dir = output_dir.joinpath(run_time)
    create_directory(output_dir)
    return output_dir

def save_to_file(curr_output_json, output_dir, i):
    # r = reward_cumsum
    # # r = reward_cumsum.tolist()
    # curr_output_json = {"no_agents": no_agents,
    #                     "reward": r}
    # i += 3000
    curr_output_json = recursive_conversion(curr_output_json)
    with open(os.path.join(output_dir, 'data_{:05}.json'.format(i)), 'w') as json_file:
        json.dump(curr_output_json, json_file, indent=0, separators=(',', ':'))


def save_params_to_file(params: dict, path: Path):
    _params = recursive_conversion(params)
    with path.open('w') as rf:
        json.dump(_params, rf, indent=0)

def custom_log_creator(results_dir):
    def logger_creator(config):
        return UnifiedLogger(config, str(results_dir), loggers=None)

    return logger_creator



def figure_configuration_ieee_standard():
    # IEEE Standard Figure Configuration - Version 1.0

    # run this code before the plot command

    #
    # According to the standard of IEEE Transactions and Journals:

    # Times New Roman is the suggested font in labels.

    # For a singlepart figure, labels should be in 8 to 10 points,
    # whereas for a multipart figure, labels should be in 8 points.

    # Width: column width: 8.8 cm; page width: 18.1 cm.

    # width & height of the figure
    k_scaling = 0.85
    # scaling factor of the figure
    # (You need to plot a figure which has a width of (8.8 * k_scaling)
    # in MATLAB, so that when you paste it into your paper, the width will be
    # scalled down to 8.8 cm  which can guarantee a preferred clearness.

    k_width_height = 1.3#1.3  # width:height ratio of the figure

    # fig_width = 17.6/2.54 * k_scaling
    # fig_width = 13.2/2.54 * k_scaling       #fig 7
    fig_width = 8.8/2.54 * k_scaling      # clos N M
    fig_height = fig_width / k_width_height

    # ## figure margins
    # top = 0.5  # normalized top margin
    # bottom = 3	# normalized bottom margin
    # left = 4	# normalized left margin
    # right = 1.5  # normalized right margin

    # #fig 7
    # params = {'axes.labelsize': 18,  # fontsize for x and y labels (was 10)
    #           'axes.titlesize': 22,
    #           'font.size': 16,  # was 10
    #           'legend.fontsize': 14,  # was 10
    #           'xtick.labelsize': 16,
    #           'ytick.labelsize': 16,
    #           'figure.figsize': [fig_width, fig_height],
    #           'font.family': 'serif',
    #           'font.serif': ['Times New Roman'],
    #           'lines.linewidth': 3,
    #           'axes.linewidth': 1,
    #           'axes.grid': True,
    #           'savefig.format': 'pdf',
    #           'axes.xmargin': 0,
    #           'axes.ymargin': 0,
    #           'savefig.pad_inches': 0.04,
    #           'legend.markerscale': 0.5,
    #           'savefig.bbox': 'tight',
    #           'lines.markersize': 2,
    #           'legend.numpoints': 3,
    #           'legend.handlelength': 2, #was 3.5
    #           'text.usetex': True
    #           }

    #fig 8 and 9
    params = {'axes.labelsize': 14,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 18,
              'font.size': 14,  # was 10
              'legend.fontsize': 12,  # was 10
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif',
              'font.serif': ['Times New Roman'],
              'lines.linewidth': 3,
              'axes.linewidth': 1,
              'axes.grid': True,
              'savefig.format': 'pdf',
              'axes.xmargin': 0,
              'axes.ymargin': 0,
              'savefig.pad_inches': 0.04,
              'legend.markerscale': 0.3,
              'savefig.bbox': 'tight',
              'lines.markersize': 1,
              'legend.numpoints': 1,
              'legend.handlelength': 2, #was 3.5
              'text.usetex': True
              }

    matplotlib.rcParams.update(params)
