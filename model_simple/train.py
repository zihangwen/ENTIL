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

    # graph_file = "_"
    # technique = "gradient"
    # n_episodes = 1000
    # obj = 5
    # sound = 5
    # c_entropy = 0.5
    # out_file = "/home/zihangw/socialAI/results_simple/paper_regular_100_10_triangle/test"

    select_period = 100

    game_config = {"num_samples" : 100}

    training_config = {"technique" : technique,
                       "c_entropy" : c_entropy,
                       "select_period" : select_period}
    
    # hyp_pool = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]

    sol = Solution(obj, sound)
    sol.training_config = training_config
    sol.game_config = game_config
    
    data = []
    t_start = time.time()
    runtime = 0

    for i_epoch in range(0, n_episodes, select_period):
        # sol.hyper_config = {"sigma" : np.random.choice(hyp_pool)}
        sol.hyper_config = {"sigma" : 0.01}
        payoff, reward = sol.train()

        for i in range(select_period):
            p = payoff[i]
            r = reward[i]
            data += [(i_epoch + 1 * i, p, r)]
            # print(*data[-1])
    
    print("# runtime: ", runtime + time.time() - t_start)
    np.savetxt(out_file + "_%s.txt" %technique, data, fmt='%g')