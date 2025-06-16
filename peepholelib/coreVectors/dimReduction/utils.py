# python stuff
from math import floor

# torch stuff
import torch

def unroll_conv2d_activations(acts, layer):
    '''
    Unroll activations of a `torch.nn.Conv2d` layer. Used during the svd projection `coreVectors.dimReduction.svd.conv2d_kernel_svd_projection()`
    Input activations have shape `[ns, cin, ih, iw]`, unrolled activations have shape `[ns, cin*kh*kw, oh*ow]`, where `cin` is the number of input channels, and `ih, iw, kh, kw, oh, ow` are the activations, kernel and output hight and width.

    Args:
    - acts (torch.tensor): batched activations
    - layer (torch.nn.Conv2d): layer (for getting padding, stride, dilation and kernel shapes

    Returns:
    - ui (torch.tensor): unrolled activations 
    - oh (int): output height
    - ow (int): output width
    '''

    if not isinstance(layer, torch.nn.Conv2d):
        raise RuntimeError('Input layer should be a torch.nn.Conv2D one')

    if layer.groups != 1:
        raise RuntimeError('Groups not implemented. Fell free to submit a PR.')


    weight = layer.weight        
    bias = layer.bias
    groups = layer.groups
    pad_mode = layer.padding_mode
    
    # kernel offsets
    kh, kw = weight.shape[2], weight.shape[3]
    ph, pw = layer.padding
    sh, sw = layer.stride
    dh, dw = layer.dilation
    
    ih, iw = acts.shape[2], acts.shape[3] 
    oh = int(floor((ih+2*ph - dh*(kh - 1) -1)/sh + 1))
    ow = int(floor((iw+2*pw - dw*(kw - 1) -1)/sw + 1))

    ui = torch.nn.functional.unfold(
            acts,
            kernel_size = (kh, kw),
            dilation = (dh, dw),
            padding = (ph, pw),
            stride = (sh, sw)
            )

    if not layer.bias == None:
        ones = torch.ones(ui.shape[0], 1, oh*ow).to(ui.device)
        ui = torch.hstack((ui, ones))

    return ui 
