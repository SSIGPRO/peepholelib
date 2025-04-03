# torch stuff
import torch
import torchvision
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.functional import pad
    
def svd_Linear(act_data, reduct_m, device):

    reduct_m = reduct_m.to(device)
    n_act = act_data.shape[0]
    acts_flat = act_data.flatten(start_dim=1)
    ones = torch.ones(n_act, 1)
    _acts = torch.hstack((acts_flat, ones)).to(device)
    cvs = (reduct_m@_acts.T).T
    cvs = cvs.cpu()

    return cvs

def svd_Linear_ViT(act_data, reduct_m, device):

    reduct_m = reduct_m.to(device)
    n_act = act_data.shape[0]
    act_data = act_data[:, 0, :] # take 0-th patch
    
    acts_flat = act_data.flatten(start_dim=1)
    ones = torch.ones(n_act, 1)
    _acts = torch.hstack((acts_flat, ones)).to(device)
    cvs = (reduct_m@_acts.T).T
    cvs = cvs.cpu()

    return cvs

def svd_Conv2D(act_data, reduct_m, layer, device):
    reduct_m = reduct_m.to(device)
    pad_mode = layer.padding_mode if layer.padding_mode != 'zeros' else 'constant'
    padding = _reverse_repeat_tuple(layer.padding, 2) 
    n_act = act_data.shape[0]
    acts_pad = pad(act_data, pad=padding, mode=pad_mode)

    acts_flat = acts_pad.flatten(start_dim=1)
    if layer.bias is None:
        _acts = acts_flat.to(device)
    else:
        ones = torch.ones(n_act, 1)
        _acts = torch.hstack((acts_flat, ones)).to(device)
    cvs = (reduct_m@_acts.T).T
    cvs = cvs.cpu()
        
    return cvs
