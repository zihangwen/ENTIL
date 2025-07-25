import sys
import time
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import gymnasium as gym
import random

from algo.agent import *
from algo.ppo import PPO
from algo.wrapper import *

if __name__ == "__main__":
    graph_file = "/Users/sumat/Downloads/3-r.txt"
    technique = "gradient"
    n_episodes = 1000
    obj = 5
    sound = 5
    out_file = "/Users/sumat/Downloads/ENTIL-benchmark 2/workspace/algo/3-r_graph_ppo"
    out_file += "_%s" %technique


    print("# cpus : ", os.cpu_count())
    print("# torch interloop threads : ", torch.get_num_interop_threads())
    print("# torch intraloop threads : ", torch.get_num_threads() )

    ############################################################################################
    select_period = 100

    game_config = {
        'T_max': 1000,
        'N_games': 4,
        'game_name': 'HalfCheetah-v5',
    }

    training_config = {
        'gamma': 0.99,
        'epsilon': 0.2,
        'lam': 0.95,
        'target_kl': 0.01,
        'train_a_iters': 80,
        'train_v_iters': 80,
    }

    sol_pool_size = 20
    max_clones = 20
   # hyp_pool = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2] #exploration choices

    select_period = 100


    sol_pool = []

    temp_env = gym.make_vec(
        game_config['game_name'],
        num_envs=game_config['N_games'],
        vectorization_mode="sync",
    )
    n_obs = temp_env.observation_space.shape[1]
    n_action = temp_env.action_space.shape[1]
    action_type = temp_env.action_space.__class__.__name__
    temp_env.close()  

    cfg = AgentConfig(n_input_actor=n_obs, n_input_critic=n_obs, n_action=n_action,
                    action_type=action_type, n_hidden=64)

    for i in range(sol_pool_size):
        seed = 1000 + i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = gym.make_vec(
            game_config['game_name'],
            num_envs=game_config['N_games'],
            vectorization_mode="sync"
        )
        env.reset(seed=seed)

        agent = AgentModule(cfg)
        algo = PPO(env, agent)
        algo.training_config = training_config
        algo.game_config = game_config
        algo.idx = i
        sol_pool.append(algo)


    G = load_G(graph_file)
    pop = Pop_wrapper(Population(G), sol_pool_size,0, max_clones)

    data = []

    print("# Starting freq: ", np.array(pop.freq_current))
    data = []
    t_start = time.time()
    runtime = 0

    for epoch in range(0, n_episodes, select_period):
        print(epoch)
        rewards = []
        for sol_idx in pop.sol_current:
            agent = sol_pool[sol_idx]
           # sigma = hyp_pool[hyp_idx]

            logger = agent.train(select_period)

            reward_trace = logger._logs["AverageEpRet"]   # list, len == select_period
            rewards.append(reward_trace)

        # 3. stack → [T, clones]
        rewards_train = np.stack(rewards).T
        freq    = np.array(pop.freq_current)
        fitness = rewards_train[-1]

        for i in range(select_period):
            r = rewards_train[i]
            data.append((epoch + i, (r * freq).sum(), r.max()))

            # print(*data[-1])
            '''
            data collects: episode index
            population wide-sum: payoffs for current ep * active copies of each clone
            highest performing clone: p.max()
            '''
        pop.update(fitness, t_gen=10)

    print("# runtime: ", runtime + time.time() - t_start)
    np.savetxt(out_file + ".txt", data, fmt='%g')

