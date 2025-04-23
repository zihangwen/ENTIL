# %%
import sys
import time
import os

import numpy as np
# import matplotlib.pyplot as plt
import pickle

import torch 
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym

from algo.wrapper import *
from algo.agent import AgentConfig, AgentModule
from algo.ppo import PPO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
n_games = 16
c_entropy = 0.5

# %%
game_config = {
    'T_max' : 1000,
    'N_games' : n_games,
    'game_name' : "HalfCheetah-v5",
}

training_config = {
    'gamma' : 0.95,
    'std_u' : 0.1,
    'epsilon' : 0.1,
    'c_entropy': c_entropy
}

# %%
envs = gym.make_vec(
    game_config['game_name'], num_envs=game_config['N_games'], vectorization_mode="sync"
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
algo.hyper_config = {'std_c' : 0.1}
# %%
reward_training = algo.train(30000)

# %%
with open("reward_training.pkl", "wb") as f:
    pickle.dump(reward_training, f)
# %%
