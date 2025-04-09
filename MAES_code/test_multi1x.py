import sys
import time
import os

NCORE = "1"
MAX_RUNTIME = 2

os.environ["OMP_NUM_THREADS"] = NCORE
os.environ["OPENBLAS_NUM_THREADS"] = NCORE
os.environ["MKL_NUM_THREADS"] = NCORE
os.environ["VECLIB_MAXIMUM_THREADS"] = NCORE
os.environ["NUMEXPR_NUM_THREADS"] = NCORE

import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
#import torch.multiprocessing as mp
import multiprocessing as mp

from previous.world import *
from previous.util import *
from previous.wrapper import *
from previous.model import *

def make_checkpoint(sol_pool):
    checkpoint = dict()
    for i, sol in enumerate(sol_pool):
        checkpoint["sol%d" % i] = sol.agent.state_dict()
        checkpoint["opt%d" % i] = sol.optimizer.state_dict()
        checkpoint["norm_a%d" % i] = sol.norm_advantage
        checkpoint["norm_r%d" % i] = sol.norm_reward
    return checkpoint

# PPO
if __name__ == '__main__':
############################################################################################
    graph_file = sys.argv[1]
    n_episodes = int(sys.argv[2])
    n_games = int(sys.argv[3])
    c_entropy = float(sys.argv[4])
    internalize = (sys.argv[5] == "True")
    out_file = sys.argv[6]

    # graph_file = "MAES/graphs/wm.txt"
    # n_episodes = 30000
    # n_games = 1024
    # c_entropy = 0.5
    # internalize = False
    # out_file = "MAES/results/wm05_comm0"

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
    try:
        print("temp_dir/%s.checkpoint"% out_file.split("/")[-1])
        checkpoint = torch.load("temp_dir/%s.checkpoint"% out_file.split("/")[-1])
        for i_sol, sol in enumerate(sol_pool):
            sol.agent.load_state_dict(checkpoint["sol%d" % i_sol])
            sol.optimizer.load_state_dict(checkpoint["opt%d" % i_sol])
            sol.norm_advantage = checkpoint["norm_a%d" % i_sol]
            sol.norm_reward = checkpoint["norm_r%d" % i_sol]
            
        pop = checkpoint["pop_wrapper"]
        starting_epoch = checkpoint['epoch']
        data = checkpoint['data']
        runtime = checkpoint['runtime']
        print("# Checkpoint loaded.")
    except:
        print("# No checkpoint detected.")
        G = load_G(graph_file)
        pop = Pop_wrapper(Population(G), sol_pool_size, len(hyp_pool), max_clones)

############################################################################################
    print("# Starting freq: ", np.array(pop.freq_current))
    worker_pool = mp.Pool(processes = num_of_process)
    for i_epoch in range(starting_epoch, n_episodes, select_period):
        # Checkpoint
        if (time.time() - t_start) // 3600 >= MAX_RUNTIME:
            runtime += (time.time() - t_start)
            print("# Checkpoint saving at: ", (time.time() - t_start) / 3600)
            print("# Current freq: ", np.array(pop.freq_current))
            checkpoint = make_checkpoint(sol_pool)
            checkpoint["pop_wrapper"] = pop
            checkpoint['epoch'] = i_epoch
            checkpoint['data'] = data
            checkpoint['runtime'] = runtime
            torch.save(checkpoint, "temp_dir/%s.checkpoint" % out_file.split("/")[-1])
            
            worker_pool.close()
            worker_pool.join()
            sys.exit(85)
        
        # Training and evaluation
        procs = []
        for sol_idx, hyp_idx in pop.sol_current:
            sol = sol_pool[sol_idx]
            sol.hyper_config = {'std_c' : hyp_pool[hyp_idx]}
            procs += [worker_pool.apply_async(sol.wrapper, (select_period, eval_period))]
            
        results = []
        for proc in procs:
            temp = proc.get()
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
            
    worker_pool.close()
    worker_pool.join()

    print("# runtime: ", runtime + time.time() - t_start)
    np.savetxt(out_file + ".txt", data, fmt='%g')
    np.savetxt(out_file + "_c.txt", pop.visit_cnts, fmt='%g')
    np.savetxt(out_file + "_f.txt", pop.visit_fits, fmt='%g')

    checkpoint = make_checkpoint(sol_pool)
    checkpoint["pop_wrapper"] = pop
    checkpoint['epoch'] = i_epoch
    checkpoint['data'] = data
    checkpoint['runtime'] = runtime
    torch.save(checkpoint, out_file + "_final.checkpoint")