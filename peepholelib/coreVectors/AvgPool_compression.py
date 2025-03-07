# General python stuff
from tqdm import tqdm

# torch stuff
import torch
import torchvision
from torchvision.models.vision_transformer import VisionTransformer
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.functional import pad

def AvgPool_reduction(cvs_dl, act_dl, lk, model, device, verbose, **kwargs):

    '''
    Avgpool_reduction computes the reduction of the convultional layers by 
    performing an avgerage channelwise

    input:
    - cvs_dl: dataset corevectors to compute
    - act_dl: dataset of activations
    - _layers_to_save: list of target_layers we are working on
    - parser_kwargs: dictionary that contains the matrices used for 
                     the corevectors extraction
    output:
    - cvs_dl: dataset containing corevectors
    '''

    layer = model._target_layers[lk]
    
    if verbose: print(f'\n ---- Getting corevectors for {lk}\n')
        
    
    if isinstance(layer, torch.nn.Linear):
        for cvs_data, act_data in tqdm(zip(cvs_dl, act_dl), disable=not verbose, total=len(cvs_dl)):
            # check if its ViT model
            if isinstance(model._model, VisionTransformer):
                if len(act_data['in_activations'][lk].shape) == 3:
                    acts = act_data['in_activations'][lk][:, 0, :] # take 0-th patch
                elif len(act_data['in_activations'][lk].shape) == 2:
                    acts = act_data['in_activations'][lk]
            else:
                acts = act_data['in_activations'][lk]
            
            n_act = act_data['in_activations'][lk].shape[0]
            acts_flat = acts.flatten(start_dim=1)
            ones = torch.ones(n_act, 1)
            _acts = torch.hstack((acts_flat, ones)).to(device)
            phs = (reduct_m@_acts.T).T
            cvs_data['coreVectors'][lk] = phs.cpu()
            
    if isinstance(layer, torch.nn.Conv2d):
        pad_mode = layer.padding_mode if layer.padding_mode != 'zeros' else 'constant'
        padding = _reverse_repeat_tuple(layer.padding, 2) 
        for cvs_data, act_data in tqdm(zip(cvs_dl, act_dl), disable=not verbose, total=len(cvs_dl)):
            n_act = act_data['in_activations'][lk].shape[0]
            acts = act_data['in_activations'][lk]
            acts_pad = pad(acts, pad=padding, mode=pad_mode)

            acts_flat = acts_pad.flatten(start_dim=1)
            if layer.bias is None:
                _acts = acts_flat.to(device)
            else:
                ones = torch.ones(n_act, 1)
                _acts = torch.hstack((acts_flat, ones)).to(device)
            cvs = (reduct_m@_acts.T).T
            cvs_data['coreVectors'][lk] = cvs.cpu() 
            
    return cvs_dl

def SVD_size(lk, **kwargs):
    reduct_matrices = kwargs['SVD']
    reduct_m = reduct_matrices_from_svds(reduct_matrices, lk)
    return reduct_m.shape[0]