import torch

def dummy_dim_reduction(act_data, size):
    n = act_data.shape[0]
    r = torch.zeros(n, size)
    for s, d in enumerate(act_data):
        for i in range(size):
            r[s, i] = (i+1)*torch.norm(d)
    return r
