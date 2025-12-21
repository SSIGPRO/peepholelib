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
        verbose = kwargs.get('verbose', False)
                                                      
        # create folder
        path.mkdir(parents=True, exist_ok=True)
        file_path = path/layer

        # get ref for the layer
        _layer = model._target_modules[layer]
        device = model.device

        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._svd = torch.load(file_path)
        else: 
            if not isinstance(_layer, torch.nn.Conv2d):
                raise RuntimeError("Only Conv2D is suported") 

            # computation
            uw = _layer.weight.flatten(start_dim=1, end_dim=-1)
            if not _layer.bias == None:
                uw = torch.hstack([uw, _layer.bias.view(-1,1)])
            
            uw = uw.to(device)
            U, s, Vh = torch.svd_lowrank(uw, q=q)
            
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
        Output shape is `[ns, cv_dim]`, trimmed corevectors
    
        Args:
            cvs (TensorDict): Batch from TensorDict for corevectors inside `peepholelib.CoreVectors` class.
            dss (TensorDict): Batch from TensorDict for dataset inside `peepholelib.CoreVectors` class
            cv_dim (int): desired dimension of corevector
            label_key (str): key to get labels from
    
        Returns:
            tcvs (torch.tensor): Trimmed corevectors and correspective labels
            labels (torch.tensor): Labels from datasate for the samples. Only returned if `dss` is given
        """
    
        cvs = kwargs['cvs']
        dss = kwargs.get('dss', None)
        cv_dim = kwargs['cv_dim']
        label_key = kwargs.get('label_key', 'label') 

        # trim corevectors on the last dimension
        tcvs = cvs[...,0:cv_dim]
                                                              
        ret = tcvs if dss == None else (tcvs, dss[label_key])
        return ret 

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
