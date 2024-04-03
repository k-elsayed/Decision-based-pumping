import subprocess
import multiprocessing
from pathlib import Path
import os

if __name__ == '__main__':
    child_processes = []

    all_dir = [
    'Results/QE_Pois_PPO/0322_135255/best_result/checkpoint_523'
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
        p = subprocess.Popen(['python', '-m', 'testing_trainer.policy_heatmap', curr_dir])
        child_processes.append(p)
        if len(child_processes) > num_cores-1:
            for p in child_processes:
                p.wait()
            child_processes = []

    for p in child_processes:
        p.wait()
