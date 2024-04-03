import subprocess
import multiprocessing
from pathlib import Path
import os

if __name__ == '__main__':
    child_processes = []


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
    # assert(R1_arr_rate + R2_arr_rate < 1 / delta_t_g)
    # F_E_0 = 0.75
    # F_R_1 = F_E_0 + 0.2
    # F_R_2 = F_E_0 - 0.2
    # T = 0.05  # time horizon in ms
    # trained_policy = 'QE_Pois'
    # sample_size = 10000
    # ppo_actions_discrete = False
    all_dir = [
    'Results/QE_Pois_PPO/0327_020117/best_result/checkpoint_304'
    ]



    num_cores = multiprocessing.cpu_count()

    for curr_dir in all_dir:

        chk = curr_dir.split('checkpoint_')[1]
        checkpoint_path = Path(curr_dir)
        chkpnt_number = chk.lstrip('0')
        new_folder_name = 'checkpoint_' + chkpnt_number
        new_name = checkpoint_path.parent.joinpath(new_folder_name)
        normalized_path1 = os.path.abspath(new_name)
        normalized_path2 = os.path.abspath(curr_dir)
        if not os.path.exists(normalized_path1):
            if normalized_path1 != normalized_path2:
                os.rename(checkpoint_path, new_name)
                curr_dir = str(new_name)
            else:
                print("Please rename path")
        p = subprocess.Popen(['python', '-m', 'testing_trainer.evaluate_policy', curr_dir])
        child_processes.append(p)
        if len(child_processes) > num_cores-1:
            for p in child_processes:
                p.wait()
            child_processes = []

    for p in child_processes:
        p.wait()
