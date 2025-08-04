# %%
# import sys
import time
# import os
from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt
# import pickle

import torch 
# import torch.nn as nn
# import torch.optim as optim

import gymnasium as gym

# from algo.wrapper import *
from algo.agent import AgentConfig, AgentModule
from algo.ppo import PPO
from arguments import get_args

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start_time = time.time()

# %%
game_name = "HalfCheetah-v5"
n_games = 4
T_max = 1000
n_epochs = 250
seed = 0
out_dir = Path(f"data/ENTIL/{game_name}")

gamma = 0.99
epsilon = 0.2
lam = 0.97
target_kl = 0.01
train_a_iters = 80
train_v_iters = 80
n_hidden = 64

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
# np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

envs = gym.make_vec(
    game_config['game_name'], num_envs=game_config['N_games'], vectorization_mode="sync"
)
envs.reset(seed=seed)

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
logger.save_logs(out_dir / "logger.pkl")
torch.save(algo.agent.state_dict(), out_dir / "ppo_agent.pt")

# %%
# os.makedirs(out_dir, exist_ok=True)

# with open(out_dir / "reward_training.pkl", "wb") as f:
#     pickle.dump(reward_training, f)

# %%
print(f"--- {time.time() - start_time} seconds ---")
