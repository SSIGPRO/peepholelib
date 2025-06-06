# python stuff
from math import floor

# torch stuff
import torch

def unroll_activations(acts, layer):
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

    return ui, oh, ow 
