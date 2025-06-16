import scipy
import numpy as np
import torch


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    x_i, _, x_k = x.shape
    out = np.zeros_like(x)
    for i in range(x_i):
        for k in range(x_k):
            out[i,:,k] = scipy.signal.lfilter([1], [1, float(-discount)], x[i,:,k].numpy()[::-1], axis=0)[::-1]
    return torch.as_tensor(out)

