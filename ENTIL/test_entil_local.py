import sys
import time
import os
from pathlib import Path

NCORE = "1"
MAX_RUNTIME = 1

os.environ["OMP_NUM_THREADS"] = NCORE
os.environ["OPENBLAS_NUM_THREADS"] = NCORE
os.environ["MKL_NUM_THREADS"] = NCORE
os.environ["VECLIB_MAXIMUM_THREADS"] = NCORE
os.environ["NUMEXPR_NUM_THREADS"] = NCORE

import pickle
import numpy as np
import torch 
# import torch.nn as nn
# import torch.optim as optim
#import torch.multiprocessing as mp
import multiprocessing as mp

import gymnasium as gym

# from algo.wrapper import *
from algo.agent import AgentConfig, AgentModule
from algo.ppo import PPO
from arguments import get_args

# from previous.world import *
# from previous.util import *
from algo.wrapper import (
    load_G,
    Pop_wrapper,
    Population,
)
# from previous.model import *

# def make_checkpoint(sol_pool):
#     checkpoint = dict()
#     for i, sol in enumerate(sol_pool):
#         checkpoint["sol%d" % i] = sol.agent.state_dict()
#         checkpoint["opt%d" % i] = sol.optimizer.state_dict()
#         checkpoint["norm_a%d" % i] = sol.norm_advantage
#         checkpoint["norm_r%d" % i] = sol.norm_reward
#     return checkpoint

# PPO
if __name__ == '__main__':
############################################################################################
    game_name = "HalfCheetah-v5"
    n_games = 4
    T_max = 1000
    n_epochs = 250
    seed = 0
    out_dir = Path(f"data/ENTIL_graph/{game_name}")

    gamma = 0.99
    epsilon = 0.2
    lam = 0.97
    target_kl = 0.01
    train_a_iters = 80
    train_v_iters = 80
    n_hidden = 64

    # ----- ----- ----- graph params ----- ----- ----- #
    graph_file = "MAES/graphs/wm.txt"
    sol_pool_size = 5
    max_clones = 5
    select_period = 10
    n_gens = 10

    # n_episodes = 30000
    # n_games = 1024
    # c_entropy = 0.5
    # internalize = False
    # out_file = "MAES/results/wm05_comm0"

    num_of_process = 5

    # print("# cpus : ", os.cpu_count() )
    # print("# torch interloop threads : ", torch.get_num_interop_threads() )
    # print("# torch intraloop threads : ", torch.get_num_threads() )
    # print("# internalize: %s, graph file: %s, entropy: %g" % (internalize, graph_file, c_entropy))
    
############################################################################################
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

############################################################################################
    sol_pool = []
    for sol_idx in range(sol_pool_size):
        # scenario = Scenario()
        # world = scenario.make_world(n_games, game_config['N_agents'], game_config['N_landmarks'], internalize)
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

        algo = PPO(envs, agent)
        algo.training_config = training_config
        algo.game_config = game_config
        
        algo.idx = sol_idx
        sol_pool += [algo]

############################################################################################
    # hyp_pool = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    hyp_pool = [None]
    starting_epoch = 0
    data = []
    t_start = time.time()
    runtime = 0
    
    # try:
    #     print("temp_dir/%s.checkpoint"% out_file.split("/")[-1])
    #     checkpoint = torch.load("temp_dir/%s.checkpoint"% out_file.split("/")[-1])
    #     for i_sol, sol in enumerate(sol_pool):
    #         sol.agent.load_state_dict(checkpoint["sol%d" % i_sol])
    #         sol.optimizer.load_state_dict(checkpoint["opt%d" % i_sol])
    #         sol.norm_advantage = checkpoint["norm_a%d" % i_sol]
    #         sol.norm_reward = checkpoint["norm_r%d" % i_sol]
            
    #     pop = checkpoint["pop_wrapper"]
    #     starting_epoch = checkpoint['epoch']
    #     data = checkpoint['data']
    #     runtime = checkpoint['runtime']
    #     print("# Checkpoint loaded.")
    # except:
    #     print("# No checkpoint detected.")
    #     G = load_G(graph_file)
    #     pop = Pop_wrapper(Population(G), sol_pool_size, len(hyp_pool), max_clones)
    G = load_G(graph_file)
    pop = Pop_wrapper(Population(G), sol_pool_size, len(hyp_pool), max_clones)

############################################################################################
    print("# Starting freq: ", np.array(pop.freq_current))
    # worker_pool = mp.Pool(processes = num_of_process)
    for i_epoch in range(starting_epoch, n_epochs, select_period):
        # Checkpoint
        # if (time.time() - t_start) // 3600 >= MAX_RUNTIME:
        #     runtime += (time.time() - t_start)
        #     print("# Checkpoint saving at: ", (time.time() - t_start) / 3600)
        #     print("# Current freq: ", np.array(pop.freq_current))
        #     checkpoint = make_checkpoint(sol_pool)
        #     checkpoint["pop_wrapper"] = pop
        #     checkpoint['epoch'] = i_epoch
        #     checkpoint['data'] = data
        #     checkpoint['runtime'] = runtime
        #     torch.save(checkpoint, "temp_dir/%s.checkpoint" % out_file.split("/")[-1])
            
        #     worker_pool.close()
        #     worker_pool.join()
        #     sys.exit(85)
        
        # Training and evaluation
        # procs = []
        # for sol_idx, hyp_idx in pop.sol_current:
        #     sol = sol_pool[sol_idx]
        #     sol.hyper_config = {'std_c' : hyp_pool[hyp_idx]}
        #     procs += [worker_pool.apply_async(sol.wrapper, (select_period, eval_period))]
            
        # results = []
        # for proc in procs:
        #     temp = proc.get()

        results = []
        for sol_idx, hyp_idx in pop.sol_current:
            algo = sol_pool[sol_idx]
            logger = algo.train(select_period)

            # sol.hyper_config = {'std_c' : hyp_pool[hyp_idx]}

            # temp = sol.wrapper(select_period, eval_period)
            logger.get_logs()
            temp = logger._logs["AverageEpRet"]  # list, len == select_period
            results += [temp]
            # sol = sol_pool[temp[1]]
            # sol.optimizer.load_state_dict(temp[2])
            # sol.norm_advantage = temp[3]
            # sol.norm_reward = temp[4]
            
        # Logging
        # results_train = np.stack([r[0] for r in results]).T
        # results_eval = np.stack([r[1] for r in results]).T
        results = np.array(results).T

        freq = np.array(pop.freq_current)
        fitness = results[-1]

        for i, r in enumerate(results):
            d_temp = [(i_epoch + select_period * i, (r * freq).sum(), r.max())]
            data += d_temp
            print(*d_temp)
        
        # Evolution
        if i_epoch > 0:
            pop.update(fitness, n_gens)
            
    # worker_pool.close()
    # worker_pool.join()

    print("# runtime: ", runtime + time.time() - t_start)
    # np.savetxt(out_file + ".txt", data, fmt='%g')
    # np.savetxt(out_file + "_c.txt", pop.visit_cnts, fmt='%g')
    # np.savetxt(out_file + "_f.txt", pop.visit_fits, fmt='%g')
    with open(out_dir / "data.pkl", 'wb') as f:
        pickle.dump(data, f)
