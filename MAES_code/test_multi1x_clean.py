import sys
import time
import os

NCORE = "1"
MAX_RUNTIME = 1

os.environ["OMP_NUM_THREADS"] = NCORE
os.environ["OPENBLAS_NUM_THREADS"] = NCORE
os.environ["MKL_NUM_THREADS"] = NCORE
os.environ["VECLIB_MAXIMUM_THREADS"] = NCORE
os.environ["NUMEXPR_NUM_THREADS"] = NCORE

import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim

from previous.world import *
from previous.util import *
from previous.wrapper import *
from previous.model import *

# PPO
if __name__ == '__main__':
############################################################################################
    graph_file = "MAES/graphs/wm.txt"
    n_episodes = 30000
    n_games = 1024
    c_entropy = 0.5
    internalize = False
    out_file = "MAES/results/wm05_comm0"

    num_of_process = 5

    print("# cpus : ", os.cpu_count() )
    print("# torch interloop threads : ", torch.get_num_interop_threads() )
    print("# torch intraloop threads : ", torch.get_num_threads() )
    print("# internalize: %s, graph file: %s, entropy: %g" % (internalize, graph_file, c_entropy))
    
############################################################################################
    game_config = {'T_max' : 100,
                   'N_games' : n_games,
                   'N_agents' : 2,
                   'N_landmarks' : 3}

    training_config = {'gamma' : 0.95,
                       'std_u' : 0.1,
                       'epsilon' : 0.1,
                       'c_entropy': c_entropy}

############################################################################################
    sol_pool_size = 5
    sol_pool = []
    for sol_idx in range(sol_pool_size):
        scenario = Scenario()
        world = scenario.make_world(n_games, game_config['N_agents'], game_config['N_landmarks'], internalize)
        env = SimpleEnv(world, scenario)

        sol = Solution()
        sol.env = env

        sol.training_config = training_config
        sol.game_config = game_config
        
        sol.idx = sol_idx
        sol_pool += [sol]

    hyp_pool = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    max_clones = 5

    select_period, eval_period = 100, 10
    starting_epoch = 0
    data = []
    t_start = time.time()
    runtime = 0
    
############################################################################################
    G = load_G(graph_file)
    pop = Pop_wrapper(Population(G), sol_pool_size, len(hyp_pool), max_clones)

############################################################################################
    print("# Starting freq: ", np.array(pop.freq_current))
    for i_epoch in range(starting_epoch, n_episodes, select_period):
        # Training and evaluation
        results = []
        for sol_idx, hyp_idx in pop.sol_current:
            sol = sol_pool[sol_idx]
            sol.hyper_config = {'std_c' : hyp_pool[hyp_idx]}

            temp = sol.wrapper(select_period, eval_period)
            results += [temp[0]]
            sol = sol_pool[temp[1]]
            sol.optimizer.load_state_dict(temp[2])
            sol.norm_advantage = temp[3]
            sol.norm_reward = temp[4]
            
        # Logging
        results_train = np.stack([r[0] for r in results]).T
        results_eval = np.stack([r[1] for r in results]).T

        freq = np.array(pop.freq_current)
        fitness = results_train[-1]

        for i, r in enumerate(results_eval):
            data += [(i_epoch + eval_period * i, (r * freq).sum(), r.max())]
            print(*data[-1])
        
        # Evolution
        if i_epoch > 0:
            pop.update(fitness)
            
    print("# runtime: ", runtime + time.time() - t_start)
    np.savetxt(out_file + ".txt", data, fmt='%g')
    np.savetxt(out_file + "_c.txt", pop.visit_cnts, fmt='%g')
    np.savetxt(out_file + "_f.txt", pop.visit_fits, fmt='%g')

