# torch stuff
import torch
import torchvision
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.functional import pad

# Our stuff 
from peepholelib.coreVectors.dimReduction.utils import unroll_conv2d_activations

def linear_svd_projection(**kwargs):
    '''
    Applies the SVD projection to `torch.Linear` activations. The output has shape `[ns, q]`, where `ns` is the number of samples in the batch, and `q` the SVD rank.

    Args:
    - act_data (torch.tensor): batched input activations
    - svd (dict{torch.tensor}): SVDs of Toeplitz unrolled layer's kernel (see `models.svd_fns.linear_svd()`) 
    - device (torch.device): device to perform computations

    Returns:
    - cvs (torch.tensor) = batched projected activations
    '''

    act_data = kwargs['act_data'] 
    svd = kwargs['svd'] 
    use_s = kwargs.get('use_s', False)
    device = kwargs['device'] 
    
    if use_s:
        reduct_m = (torch.diag(svd['s'])@svd['Vh']).detach().to(device)
    else:
        reduct_m = svd['Vh'].detach().to(device)

    n_act = act_data.shape[0]
    acts_flat = act_data.flatten(start_dim=1)
    ones = torch.ones(n_act, 1, device=device)
    _acts = torch.hstack((acts_flat, ones))
    print(f'reduct_m{reduct_m.shape}\n_acts.T{_acts.T.shape}')
    
    cvs = (reduct_m@_acts.T).T

    return cvs

def linear_svd_projection_ViT(**kwargs):
    '''
    Same as `linear_svd_projection()` but for ViT. In this case the activations are divided by patchs, we only consider the first patch, which is related to the classification token.
    '''
    act_data = kwargs['act_data'] 
    svd = kwargs['svd'] 
    use_s = kwargs.get('use_s', False)
    device = kwargs['device'] 

    if use_s:
        reduct_m = (torch.diag(svd['s'])@svd['Vh']).detach().to(device)
    else:
        reduct_m = svd['Vh'].detach().to(device)

    n_act = act_data.shape[0]
    act_data = act_data[:, 0, :] # take 0-th patch
    
    acts_flat = act_data.flatten(start_dim=1)
    ones = torch.ones(n_act, 1, device=device)
    _acts = torch.hstack((acts_flat, ones))
    cvs = (reduct_m@_acts.T).T

    return cvs

def conv2d_toeplitz_svd_projection(**kwargs):
    '''
    Applies the the Toeplitz SVD projection to `torch.Conv2d` activations. If `svd['Vh']` has 2 dimensions (non channel_wise) the output has shape `[ns, q]`, and if it has 3 dimensions (channel_wise) the output has shape `[ns, cout*q]`, where `ns` is the number of samples in the batch, `cout` the number of output channels from the layer, and `q` the SVD rank.

    Args:
    - act_data (torch.tensor): batched input activations
    - svd (dict{torch.tensor}): SVDs of Toeplitz unrolled layer's kernel (see `models.svd_fns.conv2d_toeplitz_svd()`) 
    - layer (torch.nn.Conv2d): layer to get padding
    - device (torch.device): device to perform computations
    
    Returns:
    - cvs (torch.tensor) = batched projected activations
    '''
    act_data = kwargs['act_data'] 
    svd = kwargs['svd'] 
    layer = kwargs['layer']
    use_s = kwargs.get('use_s', False)
    device = kwargs['device'] 

    if use_s:
        reduct_m = (torch.diag(svd['s'])@svd['Vh']).detach().to(device)
    else:
        reduct_m = svd['Vh'].detach().to(device)

    pad_mode = layer.padding_mode if layer.padding_mode != 'zeros' else 'constant'
    padding = _reverse_repeat_tuple(layer.padding, 2) 
    n_act = act_data.shape[0]
    acts_pad = pad(act_data, pad=padding, mode=pad_mode)

    acts_flat = acts_pad.flatten(start_dim=1)
    if layer.bias is None:
        _acts = acts_flat
    else:
        ones = torch.ones(n_act, 1, device=device)
        _acts = torch.hstack((acts_flat, ones))
    
    if len(reduct_m.shape) == 3:
        n_channels = reduct_m.shape[0]
        rank = reduct_m.shape[1]
        in_size = reduct_m.shape[2] 

        # concat channels for broadcasting multiplication
        _rm = (reduct_m.view(n_channels*rank, in_size)@_acts.T).T
        # restore shape - corevec for each channel, each channel with `rank` elements
        cvs = _rm.view(n_act, n_channels, rank) 
    else:
        cvs = (reduct_m@_acts.T).T

    return cvs

def conv2d_kernel_svd_projection(**kwargs):
    '''
    Applies the kernel SVD projection to `torch.Conv2d` activations. The output has shape `[ns, q, oh*ow]`, where `ns` is the number of samples in the batch, `q` the SVD rank, and `oh,ow` are the layer output image sizes.

    Args:
    - act_data (torch.tensor): batched input activations
    - svd (dict{torch.tensor}): SVDs of kernel unrolled layer's kernel (see `models.svd_fns.conv2d_kernel_svd()`) 
    - layer (torch.nn.Conv2d): layer to get padding
    - device (torch.device): device to perform computations
    
    Returns:
    - cvs (torch.tensor) = batched projected activations
    '''

    act_data = kwargs['act_data'] 
    svd = kwargs['svd'] 
    layer = kwargs['layer']
    use_s = kwargs.get('use_s', False)
    device = kwargs['device'] 
    
    if use_s:
        reduct_m = (torch.diag(svd['s'])@svd['Vh']).detach().to(device)
    else:
        reduct_m = svd['Vh'].detach().to(device)

    n_act = act_data.shape[0]
    unrolled_acts = unroll_conv2d_activations(acts=act_data, layer=layer)
    cvs = (reduct_m@unrolled_acts).transpose(1, 2)
    return cvs
