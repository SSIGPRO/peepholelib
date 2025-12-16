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

class Conv2dToeplitzSVD(DRB):
    def __init__(self, **kwargs):
        DRB.__init__(self, **kwargs)
        path = Path(kwargs['path'])
        layer = kwargs['layer']
        model = kwargs['model']
        q = kwargs.get('rank', 300)
        sample_in = kwargs.get('sample_in')
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
            # Turn on activation saving
            model.set_activations(save_input=True, save_output=False)
            
            # Dry run to get shapes
            with torch.no_grad():
                _in = sample_in.reshape((1,)+sample_in.shape).to(device)
                model(_in)
            in_shape = model._acts['in_activations'][layer].shape[1:]
            
            # computation
            if isinstance(_layer, torch.nn.Conv2d):
                W = c2s(in_shape, _layer, device=device) 
            elif isinstance(_layer, torch.nn.ConvTranspose2d):
                W = ct2s(in_shape, _layer, device=device) 
            else:
                raise RuntimeError("Only Conv2D and ConvTranspose2d are suported") 
            
            U, s, Vh = torch.svd_lowrank(W, q=q)
            self._svd = {
                    'U': U,
                    's': s,
                    'Vh': Vh.T
                    }

            # Turn off activation saving
            model.set_activations(save_input=False, save_output=False)

            if verbose: print(f'saving {file_path}')
            torch.save(self._svd, file_path)
        
        # save variables used in the projection a.k.a. "__call__()"
        self.bias = _layer.bias
        self.reduct_m = self._svd['Vh'].detach().to(device)
        
        self.pad_mode = _layer.padding_mode if _layer.padding_mode != 'zeros' else 'constant'
        self.padding = _reverse_repeat_tuple(_layer.padding, 2) 

        return
            
    def __call__(self, **kwargs):
        '''
        Applies the the Toeplitz SVD projection to `torch.Conv2d` activations. The output has shape `[ns, q]`, where `ns` is the number of samples in the batch and `q` the SVD rank.

        Args:
        - act_data (torch.tensor): batched input activations
        
        Returns:
        - cvs (torch.tensor) = batched projected activations
        '''
        act_data = kwargs['act_data'] 
        n_act = act_data.shape[0]
        acts_pad = pad(act_data, pad=self.padding, mode=self.pad_mode)
        acts_flat = acts_pad.flatten(start_dim=1)

        if self.bias is None:
            _acts = acts_flat
        else:
            ones = torch.ones(n_act, 1, device=acts_flat.device)
            _acts = torch.hstack((acts_flat, ones))
        
        cvs = (self.reduct_m@_acts.T).T

        return cvs
    
    def parser(self, **kwargs):
        """
        Trims corevectors obtained with `coreVectors.dimReduction.svds.conv2d_toeplitz_svd.Conv2dToeplitzSVD.
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
        cv_dim = kwargs['cv_dim']
        dss = kwargs.get('dss', None)
        label_key = kwargs.get('label_key', 'label') 

        # trim corevectors on the last dimension
        tcvs = cvs[...,0:cv_dim]

        ret = tcvs if dss == None else (tcvs, dss[label_key])
        return ret 

def c2s(input_shape, layer, device='cpu', verbose=False, warns=True):
    if not isinstance(layer, torch.nn.Conv2d):
        raise RuntimeError('Input layer should be a torch.nn.Conv2D one')

    weight = layer.weight
    bias = layer.bias
    groups = layer.groups
    stride = layer.stride
    padding = layer.padding
    dilation = layer.dilation

    if dilation != (1,1):
        raise RuntimeError('This functions does not account for dilation, if you extendent it, please send us a PR ;).')
    
    if padding != (0, 0):
        input_shape = input_shape[0:1] + torch.Size([x+2*y for x, y in zip(input_shape[-2:], padding)])
        if warns: warn('Do not forget to pad your input accoding to the Conv2d padding. Deactivate this warning passing warns=False as argument.', stacklevel=2)

    Cin, Hin, Win = input_shape
    Cout = weight.shape[0]
    Hk = weight.shape[2]
    Wk = weight.shape[3]
    kernel = weight
    kernel_size = Hk*Wk
    input_size = Hin*Win
    
    # divide the input channels and output channels in groups, if groups > 1
    if Cin % groups != 0: raise RuntimeError("Cin must be divisible by groups")
    if Cout % groups != 0: raise RuntimeErrot("Cout must be divisible by groups")
    
    Cin_g = Cin // groups  
    Cout_g = Cout // groups 

    Hout = int(floor((Hin - dilation[0]*(Hk - 1) -1)/stride[0] + 1))
    Wout = int(floor((Win - dilation[1]*(Wk - 1) -1)/stride[1] + 1))
    output_size = Hout*Wout
    
    # getting columns
    cols = torch.zeros(Cout*output_size, Cin*kernel_size + (0 if bias is None else 1), dtype=torch.int)
    data = torch.zeros(Cout*output_size, Cin*kernel_size + (0 if bias is None else 1))
    
    base_row = torch.zeros(Cin_g*kernel_size, dtype=torch.int)
    for cin in range(Cin_g):
        c_shift = cin*input_size
        for hk in range(Hk):
            h_shift = hk*Win
            for wk in range(Wk):
                idx = cin*kernel_size+hk*Wk+wk
                w_shift = wk
                base_row[idx] = c_shift+h_shift+w_shift
    
    for group_id in range(groups):
        # range to process within group
        start_index_in = group_id * Cin_g
        start_index_out = group_id * Cout_g
        end_index_in = (group_id + 1) * Cin_g
        end_index_out = (group_id + 1) * Cout_g
        group_offset = start_index_in * input_size # for correct aligment w/input 

        for cout in range(start_index_out, end_index_out): 
            k =  kernel[cout, :, :, :].flatten() 
            for ho in range(Hout):
                h_shift = ho*Win*stride[0]
                for wo in range(Wout):
                    w_shift = wo*stride[1]
                    idx = cout*output_size+ho*Wout+wo
                    shift = h_shift+w_shift+group_offset
                    cols[idx, start_index_in * kernel_size : end_index_in * kernel_size] = base_row+shift
                    data[idx, start_index_in * kernel_size : end_index_in * kernel_size] = k
                    
                    if bias is not None:
                        cols[idx, -1] = Cin * Hin * Win  
                        data[idx, -1] = bias[cout]

    shape_out = torch.Size((Cout*output_size, Cin*input_size + (0 if bias is None else 1)))
    crow = (torch.linspace(0, shape_out[0], shape_out[0]+1)*(kernel_size*Cin + (0 if bias is None else 1))).int()

    cols = cols.flatten()
    data = data.flatten()
    
    csr_mat = torch.sparse_csr_tensor(crow, cols, data, size=shape_out, device=device)

    return csr_mat 

def ct2s(input_shape, layer, device='cpu', verbose=False, warns=True):
    if not isinstance(layer, torch.nn.ConvTranspose2d):
        raise RuntimeError('Input layer should be a torch.nn.Conv2D or torch.nn.ConvTranspose2d')

    weight = layer.weight
    bias = layer.bias
    groups = layer.groups
    stride = layer.stride
    padding = layer.padding
    dilation = layer.dilation
    output_padding = layer.output_padding

    if padding != (0,0) or dilation != (1,1) or groups!=1:
        raise RuntimeError('This functions does not account for padding, dilation, and groups. If you extendent it, please send us a PR ;).')
    
    kernel = weight
    Wk = weight.shape[3]
    Hk = weight.shape[2]

    # swap in and out
    # We will compute as a normal conv and transpose later.
    # Note that the paddins are inverted here
    Hout = input_shape[1]+2*padding[0]
    Wout = input_shape[2]+2*padding[1]
    # TODO: padding is not working
    Hin=(Hout-1)*stride[0]-2*padding[0]+dilation[0]*(Hk-1)+output_padding[0]+1
    Win=(Wout-1)*stride[1]-2*padding[1]+dilation[1]*(Wk-1)+output_padding[1]+1

    Cout, Cin = kernel.shape[0], kernel.shape[1]
    kernel_size = Hk*Wk
    input_size = Hin*Win
    output_size = Hout*Wout
    
    # getting columns
    cols = torch.zeros(Cout*output_size, Cin*kernel_size, dtype=torch.int)
    data = torch.zeros(Cout*output_size, Cin*kernel_size)
    
    base_row = torch.zeros(Cin*kernel_size, dtype=torch.int)
    for cin in range(Cin):
        c_shift = cin*input_size
        for hk in range(Hk):
            h_shift = hk*Win
            for wk in range(Wk):
                idx = cin*kernel_size+hk*Wk+wk
                w_shift = wk
                base_row[idx] = c_shift+h_shift+w_shift

    for cout in range(0, Cout): 
        k =  kernel[cout, :, :, :].flatten() 
        for ho in range(Hout):
            h_shift = ho*Win*stride[0]
            for wo in range(Wout):
                w_shift = wo*stride[1]
                idx = cout*output_size+ho*Wout+wo
                shift = h_shift+w_shift
                cols[idx, 0:Cin*kernel_size] = base_row+shift
                data[idx, 0:Cin*kernel_size] = k
                    
    shape_out = torch.Size((Cout*output_size+(0 if bias is None else 1), Cin*input_size))
    crow = (torch.arange(shape_out[0])*kernel_size*Cin).int()
    if bias is not None:
        crow = torch.hstack((crow, torch.tensor([crow[-1]+Cin*input_size])))

        cols = cols.flatten()
        data = data.flatten()

        cols = torch.hstack((cols, torch.arange(Cin*Hin*Win))) 
        for cin in range(0, Cin): 
            data = torch.hstack((data, bias[cin].cpu()*torch.ones(Hin*Win))) 

    ret = torch.sparse_csr_tensor(crow, cols, data, size=shape_out, device=device)
    ret = ret.t()
    return ret 
