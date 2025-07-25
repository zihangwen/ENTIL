# %%
import sys
import time
import os
from pathlib import Path

import numpy as np
# import matplotlib.pyplot as plt
import pickle

import torch 
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym

# from algo.wrapper import *
from algo.agent import AgentConfig, AgentModule
from algo.ppo import PPO
# from arguments import get_args

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# args = get_args()
# game_name = args.env_name
# n_games = args.num_envs
# entropy_coef = args.entropy_coef
# out_dir = Path(args.out_dir)
# T_max = args.T_max
# n_epochs = args.n_epochs
# gamma = args.gamma
# epsilon = args.epsilon
# lam = args.lam
# target_kl = args.target_kl
# train_a_iters = args.train_a_iters
# train_v_iters = args.train_v_iters
# n_hidden = args.n_hidden

game_name = "HalfCheetah-v5"
n_games = 4
entropy_coef = 0
out_dir_root = Path(f"../data/ENTIL/{game_name}")   # <- renamed only to avoid clash later
T_max = 1000
n_epochs = 750
gamma = 0.99
epsilon = 0.2
n_hidden = 64
lam = 0.95
target_kl = 0.01
train_a_iters = 80
train_v_iters = 80


for seed in range(100):
    print(f"\n==========  SEED {seed}  ==========")

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # %%
    game_config = {
        'T_max' : T_max,
        'N_games' : n_games,
        'game_name' : game_name,
    }

    training_config = {
        'gamma' : gamma,
        'epsilon' : epsilon,
        # 'entropy_coef': entropy_coef,
        'lam' : lam,
        'target_kl' : target_kl,
        'train_a_iters' : train_a_iters,
        'train_v_iters' : train_v_iters,
    }

    # %%
    envs = gym.make_vec(
        game_config['game_name'], num_envs=game_config['N_games'],
        vectorization_mode="sync", seed=seed    # <- pass seed to Gym
    )

    n_obs = envs.observation_space.shape[1]
    n_action = envs.action_space.shape[1]
    action_type = envs.action_space.__class__.__name__
    n_hidden = 64

    cfg = AgentConfig(
        n_input_actor = n_obs,
        n_input_critic = n_obs,
        n_action = n_action,
        action_type = action_type,
        n_hidden = n_hidden,
    )

    agent = AgentModule(cfg)

    # %%
    algo = PPO(envs, agent)
    algo.training_config = training_config
    algo.game_config = game_config
    # algo.hyper_config = {'std_c' : None} # 0.1

    # %%
    logger = algo.train(n_epochs)

    # make per-seed output directory
    out_dir = out_dir_root / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.save_logs(out_dir / "logger.pkl")
    torch.save(algo.agent.state_dict(), out_dir / "ppo_agent.pt")

    envs.close()     # tidy up before next seed loop iteration

# %%
# os.makedirs(out_dir, exist_ok=True)

# with open(out_dir / "reward_training.pkl", "wb") as f:
#     pickle.dump(reward_training, f)


# %%
