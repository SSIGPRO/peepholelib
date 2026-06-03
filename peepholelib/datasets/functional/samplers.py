import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from peepholelib.datasets.datasetWrap import DatasetWrap 

def random_subsampling(ds, perc):
    assert(isinstance(ds, DatasetWrap))

    if isinstance(perc, float):
        perc = {k: perc for k in ds.__dataset__}

    if not isinstance(perc, dict):
        raise TypeError(f'perc must be a float or a dict, got {type(perc)}')

    for k, p in perc.items():
        ds.__dataset__[k], _ = random_split(ds.__dataset__[k], [p, 1.0 - p])
    return

def dist_preserving(data, n, weights='label'):
    raise RuntimeError('DEPRECATED')

    if torch.is_tensor(weights) and len(weights.shape) == 1:
        _w = weights.float()
    elif type(weights) == str:
        _d = data[weights].detach().int()
        _l = torch.bincount(_d)
        __w = _l/_l.sum()
        _w = torch.Tensor([__w[x] for x in _d]) 
    else:
        raise RuntimeError('wrt should be an 1-dim array containing the weights for each sample index, or a string indicating the key in `data` for computing the weights')

    n = int(n)
    sampler = WeightedRandomSampler(_w, n, replacement=False)
    _dl = DataLoader(data, batch_size=n, sampler=sampler, collate_fn=lambda x: x)
    sub_sampled_data = next(iter(_dl))
    return sub_sampled_data, _w 
