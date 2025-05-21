import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from peepholelib.datasets.dataset_base import DatasetBase 

def random_subsampling(ds, perc):
    assert(isinstance(ds, DatasetBase))
    
    for k in ds._dss:
        ds._dss[k], _ = random_split(ds._dss[k], [perc, 1.0-perc])
    return 

def dist_preserving(data, n, weights='label'):
    if torch.is_tensor(weights) and len(weights.shape) == 1:
        _w = weights 
    elif type(weights) == str:
        _d = data[weights].detach().int()
        _l = torch.bincount(_d)
        __w = _l/_l.sum()
        _w = torch.Tensor([__w[x] for x in _d]) 
    else:
        raise RuntimeError('wrt should be an 1-dim array containing the weights for each sample index, or a string indicating the key in `data` for computing the weights')

    print('n', n, type(n))
    sampler = WeightedRandomSampler(_w, len(weights))
    _dl = DataLoader(data, batch_size=n, collate_fn=lambda x: x)
    sub_sampled_data = next(iter(_dl))
    return sub_sampled_data, _w 

