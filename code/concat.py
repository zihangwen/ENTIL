import numpy as np
import pandas as pd
import os
from scipy.optimize import fsolve


def load_result(param_file, num = 20):
    result_all = pd.DataFrame()
    if len(param_file.shape) == 1:
        graph_type, graph_name, s = param_file
        catfile = pd.DataFrame()
        for rep in range(num):
            try:
                vec = pd.read_csv(os.path.join(base_dir, "results_moran", graph_type, graph_name + "_%d.txt" %(rep)), sep="\t", header=None, comment='#')
                catfile = pd.concat([catfile, vec], ignore_index=True)
            except:
                print("Empty file: " + os.path.join(base_dir, "results_moran", graph_type, graph_name + "_%d.txt" %(rep)))
        N = catfile.iloc[0,0].item()
        s = catfile.iloc[0,1].item()
        runs = catfile.iloc[:,2].sum()
        counts = catfile.iloc[:,3].sum()
        pfix = counts / runs
        time = (catfile.iloc[:,3] / counts * catfile.iloc[:,4]).sum()
        result_g = pd.DataFrame([[N, s, runs, counts, pfix, time, graph_name]], columns=["N", "s", "runs", "counts", "pfix", "time", "graph_name"])
        result_all = pd.concat([result_all, result_g], ignore_index=True)
    else:
        for graph_type, graph_name, s in param_file:
            # N	s	runs	counts	time
            catfile = pd.DataFrame()
            for rep in range(num):
                try:
                    vec = pd.read_csv(os.path.join(base_dir, "results_moran", graph_type, graph_name + "_%d.txt" %(rep)), sep="\t", header=None, comment='#')
                    catfile = pd.concat([catfile, vec], ignore_index=True)
                except:
                    # print("Empty file: " + os.path.join(base_dir, "results_moran", graph_type, graph_name + "_%d.txt" %(rep)))
                    continue
            N = catfile.iloc[0,0].item()
            s = catfile.iloc[0,1].item()
            runs = catfile.iloc[:,2].sum()
            counts = catfile.iloc[:,3].sum()
            pfix = counts / runs
            time = (catfile.iloc[:,3] / counts * catfile.iloc[:,4]).sum()
            result_g = pd.DataFrame([[N, s, runs, counts, pfix, time, graph_name]], columns=["N", "s", "runs", "counts", "pfix", "time", "graph_name"])
            result_all = pd.concat([result_all, result_g], ignore_index=True)
    return result_all

base_dir = "/home/zihangw/socialAI/"
result_folder = "results_moran"

param_name = "1_param_graphs.in"
param_dir = os.path.join(base_dir, "param_moran", param_name)
output_file_name = os.path.join(base_dir, result_folder, param_name.split(".")[0] + "_concat.csv")
param_file = np.loadtxt(param_dir, dtype=str)
result_param = load_result(param_file, 20)

wellmixed_dir = os.path.join(base_dir, "param_moran", "wellmixed.in")
wellmixed_file = np.loadtxt(wellmixed_dir, dtype=str)
result_wm = load_result(wellmixed_file, 1)

# pfix_function = lambda s, N : (1 - 1/(1+s)) / (1 - 1/(1+s)**N)
p_fix_func_solve = lambda alpha, s, N, p_fix: (1 - 1/(1+alpha*s)) / (1 - 1/(1+alpha*s)**N) - p_fix

alpha_list = []
for i in range(len(result_param)):
    s = result_param.loc[i,"s"]
    N = result_param.loc[i,"N"]
    pfix = result_param.loc[i,"pfix"]
    alpha_list += [fsolve(p_fix_func_solve, 1, args = (s, N, pfix)).item()]

result_param["alpha"] = alpha_list
result_param["acc"] = result_wm["time"].item() / result_param["time"]
# result_1_param["alpha"] = 

with open(output_file_name, "w") as of:
    of.write("# ")
    of.write("\t".join(result_param.columns.tolist()))
    of.write("\n")
result_param.to_csv(output_file_name,sep="\t",header=False,index=False,mode="a")

print("File saved to: ", output_file_name)


# wm_pfix = pd.read_csv(os.path.join(base_dir, "results_moran", "wellmixed", "wellmixed_100_0.txt"), sep="\t", header=None, comment='#')
# N = wm_pfix.iloc[0,0].item()
# s = wm_pfix.iloc[0,1].item()
# runs = wm_pfix.iloc[0,2].item()
# counts = wm_pfix.iloc[0,3].item()
# time = wm_pfix.iloc[0,4].item()
# pfix = counts / runs
# graph_name = "wellmixed_100"
# alpha = 1
# acc = 1

# result_g = pd.DataFrame([[N, s, runs, counts, pfix, time, graph_name, alpha, acc]], columns=["N", "s", "runs", "counts", "pfix", "time", "graph_name", "alpha", "acc"])
# output_file_name = os.path.join(base_dir, "results_moran", "wellmixed_100_concat.csv")
# with open(output_file_name, "w") as of:
#     of.write("# ")
#     of.write("\t".join(result_g.columns.tolist()))
#     of.write("\n")
# result_g.to_csv(output_file_name,sep="\t",header=False,index=False,mode="a")
# print("File saved to: ", output_file_name)