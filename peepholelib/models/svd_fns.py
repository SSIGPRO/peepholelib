# General python stuff
from tqdm import tqdm

# torch stuff
import torch
from math import *

# our stuff
from peepholelib.models.utils import c2s, ct2s

def linear_svd(**kwargs):
    layer = kwargs['layer']
    q = kwargs['rank'] if 'rank' in kwargs else 300
    device = kwargs['device'] if 'device' in kwargs else 'cpu'

    W_ = torch.hstack((layer.weight, layer.bias.reshape(-1,1))).to(device)
    U, s, Vh = torch.svd_lowrank(W_, q)

    return U, s, Vh.T

def conv2d_toeplitz_svd(**kwargs):
    layer = kwargs['layer']
    in_shape = kwargs['in_shape']
    q = kwargs['rank'] if 'rank' in kwargs else 300
    channel_wise = kwargs['channel_wise'] if 'channel_wise' in kwargs else True
    device = kwargs['device'] if 'device' in kwargs else 'cpu'

    if isinstance(layer, torch.nn.Conv2d):
        W_ = c2s(in_shape, layer, channel_wise=channel_wise, device=device) 
    elif isinstance(layer, torch.nn.ConvTranspose2d):
        W_ = ct2s(in_shape, layer, channel_wise=channel_wise, device=device) 
    
    # same as `if channel_wise:`
    if isinstance(W_, list):
        uu, ss, vv = [], [], []
        for csr in tqdm(W_):
            _u, _s, _v = torch.svd_lowrank(csr, q=q)
            uu.append(_u)
            ss.append(_s)
            vv.append(_v.T)
        U = torch.stack(uu)
        s = torch.stack(ss)
        Vh = torch.stack(vv)
    else:
        U, s, Vh = torch.svd_lowrank(W_, q=q)
        Vh = Vh.T

    return U, s, Vh

def conv2d_kernel_svd(**kwargs):
    layer = kwargs['layer']
    q = kwargs['rank'] if 'rank' in kwargs else 300
    device = kwargs['device'] if 'device' in kwargs else 'cpu'
    uw = layer.weight.flatten(start_dim=1, end_dim=-1)
                                                       
    if not layer.bias == None:
        uw = torch.hstack([uw, layer.bias.view(-1,1)])
    
    uw = uw.to(device)
    U, s, Vh = torch.svd_lowrank(uw, q=q)
    return U, s, Vh.T
