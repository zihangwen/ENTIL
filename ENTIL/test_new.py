# %%
import sys
import time
import os

import gymnasium as gym

import torch 
import torch.nn as nn
import torch.optim as optim

from algo.model import PolicyConfig, Policy
from algo.ppo import PPOConfig, PPO
from algo.storage import RolloutStorage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
num_envs = 8

# %%
envs = gym.make_vec("Ant-v5", num_envs=num_envs, vectorization_mode="sync")

# %%
obs_shape = envs.observation_space.shape[1:]
action_shape = envs.action_space.shape[1:]
action_type = envs.action_space.__class__.__name__
if action_type == "Discrete":
    action_shape = 1
action_shape = envs.action_space.shape[1:]
action_type
envs.action_space

policy_cfg = PolicyConfig(
    num_inputs = obs_shape[0],
    num_outputs = envs.action_space.shape[1],
    action_type = envs.action_space.__class__.__name__,
    base_name = "MLPBase",
    hidden_size = 64
)
actor_critic = Policy(policy_cfg)
actor_critic.to(device)

# %%
ppo_cfg = PPOConfig()
agent = PPO(actor_critic, ppo_cfg)

# %%
rollouts = RolloutStorage(
    num_steps = 5,
    num_processes = num_envs,
    obs_shape = envs.observation_space.shape,
    action_space = envs.action_space,
    recurrent_hidden_state_size = actor_critic.recurrent_hidden_state_size
)

obs = envs.reset()
rollouts.obs[0].copy_(obs)
rollouts.to(device)

episode_rewards = deque(maxlen=10)

start = time.time()
