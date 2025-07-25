import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 


def load_datasets(logdirs, file_pattern='**/progress.txt', xaxis='TotalEnvInteracts', value='AverageEpRet'):
    all_dfs = []
    units = {}
    for logdir in logdirs:
        pattern = os.path.join(logdir, file_pattern)
        matches = glob.glob(pattern, recursive=True)
        print(f"[DEBUG] Searching `{pattern}` → found {len(matches)} files")
        for p in matches:
            df = pd.read_csv(p, sep='\t')
            if xaxis not in df.columns or value not in df.columns:
                print(f"[WARNING] Missing `{xaxis}` or `{value}` in {p}, skipping.")
                continue
            condition = os.path.basename(os.path.dirname(p))
            units.setdefault(condition, 0)
            df['Unit'] = units[condition]
            df['Condition1'] = "Spinup"  # Force label as 'Spinup'
            units[condition] += 1
            all_dfs.append(df[[xaxis, value, 'Unit', 'Condition1']])
    return all_dfs


# def convert_logger_to_df(logger, xaxis='TotalEnvInteracts', value='AverageEpRet', label='ENTIL', steps_per_epoch=1000, n_envs=4):
#     avg_rewards = np.array(logger["AverageEpRet"])
#     std_rewards = np.array(logger["StdEpRet"])

#     total_env_interacts = np.arange(len(avg_rewards)) * steps_per_epoch * n_envs
#     df = pd.DataFrame({
#         xaxis: total_env_interacts,
#         value: avg_rewards,
#         'Std': std_rewards,
#         'Unit': 0,
#         'Condition1': label
#     })
#     return [df]  # to match list format


def smooth_data(all_dfs, value='AverageEpRet', smooth=1):
    if smooth > 1:
        y = np.ones(smooth)
        for df in all_dfs:
            x = np.asarray(df[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            df[value] = smoothed_x


def plot_data(all_dfs, xaxis='TotalEnvInteracts', value='AverageEpRet', smooth=1):
    smooth_data(all_dfs, value, smooth)
    df = pd.concat(all_dfs, ignore_index=True)

    plt.figure(figsize=(10, 6))

    for condition, group in df.groupby('Condition1'):
        stats = group.groupby(xaxis)[value].agg(['mean', 'std']).reset_index()
        plt.plot(stats[xaxis], stats['mean'], label=condition)
        plt.fill_between(stats[xaxis],
                         stats['mean'] - stats['std'],
                         stats['mean'] + stats['std'],
                         alpha=0.2)

    plt.xlabel(xaxis)
    plt.ylabel(value)
    plt.title("Training Performance Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    logdirs = [
        f"/Users/sumat/Downloads/ENTIL-benchmark 2/data/ppo_Ant-v5_seed{i}"
        for i in range(10)
    ]
    xaxis = 'TotalEnvInteracts'
    value = 'AverageEpRet'
    smooth = 11

    all_dfs = load_datasets(logdirs, xaxis=xaxis, value=value)

    entil_base_dir = "../data/ENTIL/Ant-v5"
    steps_per_epoch = 1000
    n_envs = 4
    # Load ENTIL logger.pkl
    for seed in range(10):
        pkl_path = os.path.join(entil_base_dir, f"seed_{seed}", "logger.pkl")
        if not os.path.exists(pkl_path):
            print(f"[WARNING] Missing ENTIL logger for seed {seed}: {pkl_path}")
            continue
        with open(pkl_path, "rb") as f:
            logger = pickle.load(f)
        
        avg_rewards = np.array(logger["AverageEpRet"])
        std_rewards = np.array(logger["StdEpRet"])
        total_env_interacts = np.arange(len(avg_rewards)) * steps_per_epoch * n_envs

        df = pd.DataFrame({
            xaxis: total_env_interacts,
            value: avg_rewards,
            'Std': std_rewards,
            'Unit': seed,
            'Condition1': 'ENTIL'
        })

        all_dfs.append(df)

    # Plot both
    plot_data(all_dfs, xaxis=xaxis, value=value, smooth=smooth)
