from utils import save_params_to_file, get_short_datetime, create_directory
import numpy as np
from pathlib import Path

request_arrival_rates = [10000, 12000]
F_R_all = [0.95, 0.85]

c = 2 * 10 ** 8
l_km = 1
t_c = 0.001     # in s
eta_db = 0.15
eta_linear = 10 ** (eta_db * (l_km / 10))
p_g = np.exp(-eta_linear)
delta_t = (l_km * 1000) / c   # time slot
delta_t_g = delta_t/p_g
assert(sum(request_arrival_rates) < 1 / delta_t_g)
F_E_0 = 0.75

sample_size = 2000
T = sample_size*delta_t_g
run_params = {
    'request_arrival_rates': request_arrival_rates,
    'delta_t': delta_t,
    'delta_t_g': delta_t_g,
    'T': T,
    'F_E_0': F_E_0,
    'F_R_all':F_R_all,
    'eta_db': eta_db,
    'eta_linear': eta_linear,
    'l_km': l_km,
    'p_g': p_g,
    't_c': t_c,
    'c': c,
    'sample_size': sample_size,
}
dt = get_short_datetime()

results_dir = Path('../Results')
#/Parameter_Files/TwoPoisEnv')
create_directory(results_dir)
results_dir=results_dir.joinpath('Parameter_Files')
create_directory(results_dir)
results_dir=results_dir.joinpath('TwoPoisson')
create_directory(results_dir)
save_params_to_file(run_params, results_dir.joinpath('run_params_{}Pois.json'.format(len(request_arrival_rates))))  # use if you want to save datewise