import sys
import time
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from modules.model import *
from modules.wrapper import *


if __name__ == "__main__":
    graph_file = sys.argv[1]
    technique = sys.argv[2]
    n_episodes = int(sys.argv[3])
    obj = int(sys.argv[4])
    sound = int(sys.argv[5])
    c_entropy = float(sys.argv[6])
    out_file = sys.argv[7]
    out_file += "_%s" %technique

    # graph_file = "/home/zihangw/StochasticFitness/networks/paper_regular_100_10_triangle/reg_tri_100_10_0.txt"
    # technique = "gradient"
    # n_episodes = 1000
    # obj = 5
    # sound = 5
    # c_entropy = 0.5
    # out_file = "/home/zihangw/socialAI/results_simple/paper_regular_100_10_triangle/test_graph"
    # out_file += "_%s" %technique

    print("# cpus : ", os.cpu_count())
    print("# torch interloop threads : ", torch.get_num_interop_threads())
    print("# torch intraloop threads : ", torch.get_num_threads() )
    print("# technique: %s, graph file: %s, entropy: %g" % (technique, graph_file, c_entropy))

    ############################################################################################
    select_period = 100

    game_config = {"num_samples" : 100}

    training_config = {"technique" : technique,
                       "c_entropy" : c_entropy,
                       "select_period" : select_period}
    
    ############################################################################################
    sol_pool_size = 20
    max_clones = 20
    hyp_pool = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]

    sol_pool = []
    for sol_idx in range(sol_pool_size):
        sol = Solution(obj, sound)
        sol.training_config = training_config
        sol.game_config = game_config
        
        sol.idx = sol_idx
        sol_pool += [sol]
    
    G = load_G(graph_file)
    pop = Pop_wrapper(Population(G), sol_pool_size, len(hyp_pool), max_clones)

    ############################################################################################
    print("# Starting freq: ", np.array(pop.freq_current))
    data = []
    t_start = time.time()
    runtime = 0
    
    for i_epoch in range(0, n_episodes, select_period):
        payoffs = []
        rewards = []
        for sol_idx, hyp_idx in pop.sol_current:
            sol = sol_pool[sol_idx]
            sol.hyper_config = {"sigma" : hyp_pool[hyp_idx]}
            payoff, reward = sol.train()
            payoffs += [payoff]
            rewards += [reward]
        
        payoff_train = np.stack(payoffs).T
        rewards_train = np.stack(rewards).T

        freq = np.array(pop.freq_current)
        fitness = rewards_train[-1]

        for i in range(select_period):
            p = payoff_train[i]
            r = rewards_train[i]
            data += [(i_epoch + 1 * i, (p * freq).sum(), p.max(), (r * freq).sum(), r.max())]
            # print(*data[-1])
    
        pop.update(fitness, t_gen=10)

    print("# runtime: ", runtime + time.time() - t_start)
    np.savetxt(out_file + ".txt", data, fmt='%g')
    # np.savetxt(out_file + "_c.txt", pop.visit_cnts, fmt='%g')
    # np.savetxt(out_file + "_f.txt", pop.visit_fits, fmt='%g')