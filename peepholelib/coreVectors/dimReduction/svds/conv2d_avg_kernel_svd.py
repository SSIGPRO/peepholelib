# General python stuff
from warnings import warn
from pathlib import Path
from math import *

# torch stuff
import torch
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.functional import pad

# Our stuff
from ..dim_reduction_base import DimReductionBase as DRB 

class Conv2dAvgKernelSVD(DRB):
    def __init__(self, **kwargs):
        DRB.__init__(self, **kwargs)
        path = Path(kwargs['path'])
        layer = kwargs['layer']
        model = kwargs['model']
        q = kwargs.get('rank', 300)
        self.cv_dim = kwargs.get('cv_dim', None)
        verbose = kwargs.get('verbose', False)
                                                      
        # create folder
        path.mkdir(parents=True, exist_ok=True)
        file_path = path/layer

        # get ref for the layer
        _layer = model._target_modules[layer]
        device = model.device

        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._svd = torch.load(file_path, weights_only=True)
        else: 
            if not isinstance(_layer, torch.nn.Conv2d):
                raise RuntimeError("Only Conv2D is suported") 

            # computation
            uw = flatten_conv2d_weight(_layer).to(device)
            U, s, Vh = torch.svd_lowrank(uw, q=q)
            U, s, Vh = U.detach().cpu(), s.detach().cpu(), Vh.detach().cpu()
            
            self._svd = {
                    'U': U,
                    's': s,
                    'Vh': Vh.T
                    }

            if verbose: print(f'saving {file_path}')
            torch.save(self._svd, file_path)
        
        # save variables used in the projection a.k.a. "__call__()"
        self.reduct_m = self._svd['Vh'].detach().to(device)
        self.layer = _layer    

        return
            
    def __call__(self, **kwargs):
        '''
        Applies the kernel SVD projection to `torch.Conv2d` activations. The output has shape `[ns, q, oh*ow]`, where `ns` is the number of samples in the batch, `q` the SVD rank, and `oh,ow` are the layer output image sizes.

        Args:
        - act_data (torch.tensor): batched input activations
        
        Returns:
        - cvs (torch.tensor) = batched projected activations
        '''
        act_data = kwargs['act_data'] 
        n_act = act_data.shape[0]
        unrolled_acts = unroll_conv2d_activations(acts=act_data, layer=self.layer)
        cvs = (self.reduct_m@unrolled_acts).transpose(1, 2).mean(axis=1)

        return cvs

    def parser(self, **kwargs):
        """
        Trims multi kernel corevectors obtained with `coreVectors.dimReduction.svds.conv2d_avg_kernel_svd.Conv2dAvgKernelSVD`.
        Input shape is `[ns, q]`, where `ns` is the number of samples in the batch, `q` the SVD rank.
        Output shape is `[ns, self.cv_dim]`, trimmed corevectors
    
        Args:
            cvs (TensorDict): Batch from TensorDict for corevectors inside `peepholelib.CoreVectors` class.
            dss (TensorDict): Batch from TensorDict for dataset inside `peepholelib.CoreVectors` class
            label_key (str): key to get labels from
    
        Returns:
            tcvs (torch.tensor): Trimmed corevectors and correspective labels
            labels (torch.tensor): Labels from datasate for the samples. Only returned if `dss` is given
        """
    
        cvs = kwargs['cvs']
        dss = kwargs.get('dss', None)
        label_key = kwargs.get('label_key', 'label') 

        # trim corevectors on the last dimension
        tcvs = cvs[...,0:self.cv_dim]
                                                              
        ret = tcvs if dss == None else (tcvs, dss[label_key])
        return ret 

def flatten_conv2d_weight(layer):
    '''
    Flatten a Conv2d kernel into the global input-channel layout used by
    unfolded activations. For grouped convs, inactive channel blocks are
    zero-filled
    '''

    weight = layer.weight
    bias = layer.bias
    groups = layer.groups
    cin = layer.in_channels
    cout = layer.out_channels
    cin_g = weight.shape[1]
    kh, kw = weight.shape[2], weight.shape[3]
    kernel_size = kh * kw

    if cin % groups != 0:
        raise RuntimeError('Cin must be divisible by groups')
    if cout % groups != 0:
        raise RuntimeError('Cout must be divisible by groups')

    if groups == 1:
        uw = weight.flatten(start_dim=1, end_dim=-1)
    else:
        cout_g = cout // groups
        uw = torch.zeros(
                cout,
                cin * kernel_size,
                dtype=weight.dtype,
                device=weight.device,
                )

        for group_id in range(groups):
            start_in = group_id * cin_g
            end_in = (group_id + 1) * cin_g
            start_out = group_id * cout_g
            end_out = (group_id + 1) * cout_g

            group_kernel = weight[start_out:end_out].flatten(start_dim=1, end_dim=-1)
            uw[start_out:end_out, start_in * kernel_size:end_in * kernel_size] = group_kernel

    if bias is not None:
        uw = torch.hstack((uw, bias.view(-1, 1)))

    return uw

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

    weight = layer.weight        
    bias = layer.bias
    
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
