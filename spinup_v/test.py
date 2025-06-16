# %%
from mpi4py import MPI
import numpy as np

# %%
def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)

def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff

def mpi_sum(x):
    return mpi_op(x, MPI.SUM)

# %%
x = [1,2,3,4]
x = np.array(x, dtype=np.float32)
global_sum, global_n = mpi_sum([np.sum(x), len(x)])
mean = global_sum / global_n

# %%
