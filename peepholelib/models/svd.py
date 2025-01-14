# python stuff
import numpy as np
from warnings import warn
from pathlib import Path

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

def c2s(input_shape, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1,1), device='cpu', verbose=False, warns=True):

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

    Hout = int(np.floor((Hin - dilation[0]*(Hk - 1) -1)/stride[0] + 1))
    Wout = int(np.floor((Win - dilation[1]*(Wk - 1) -1)/stride[1] + 1))

    shape_out = torch.Size((Cout*Hout*Wout, Cin*Hin*Win+1))
    
    crow = (torch.linspace(0, shape_out[0], shape_out[0]+1)*(Hk*Wk*Cin+1)).int()
    nnz = crow[-1]
    
    # getting columns
    cols = torch.zeros(Cout*Hout*Wout, Cin*Hk*Wk+1, dtype=torch.int)
    data = torch.zeros(Cout*Hout*Wout, Cin*Hk*Wk+1)
    
    base_row = torch.zeros(Cin*Hk*Wk, dtype=torch.int)
    for cin in range(Cin):
        c_shift = cin*(Hin*Win)
        for hk in range(Hk):
            h_shift = hk*Win
            for wk in range(Wk):
                idx = cin*Hk*Wk+hk*Wk+wk
                w_shift = wk
                base_row[idx] = c_shift+h_shift+w_shift
        
    for cout in range(Cout): 
        k = kernel[cout]
        _d = torch.hstack((k.flatten(), bias[cout]))
        for ho in range(Hout):
            h_shift = ho*Win*stride[0]
            for wo in range(Wout):
                w_shift = wo*stride[1]
                idx = cout*Hout*Wout+ho*Wout+wo
                shift = h_shift+w_shift
                cols[idx,:-1] = base_row+shift
                data[idx] = _d

    # add bias as the last column                    
    cols[:,-1] = Cin*Hin*Win

    cols = cols.flatten()
    data = data.flatten()

    csr_mat = torch.sparse_csr_tensor(crow, cols, data, size=shape_out, device=device)
    return csr_mat 

def get_svds(self, **kwargs):
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    path = Path(kwargs['path'])
    name = Path(kwargs['name'])

    # create folder
    path.mkdir(parents=True, exist_ok=True)
    
    file_path = path/(name.name)
    if file_path.exists():
        if verbose: print(f'File {file_path} exists. Loading from disk.')
        _svds = TensorDict.load_memmap(file_path)
    else: 
        _svds = TensorDict()

    _layers_to_compute = []
    for lk in self._target_layers:
        if lk in _svds.keys():
            continue
        _layers_to_compute.append(lk)
    if verbose: print('Layers to compute SVDs: ', _layers_to_compute)
    
    for lk in _layers_to_compute:
        if verbose: print(f'\n ---- Getting SVDs for {lk}\n')
        layer = self._target_layers[lk]
        weight = layer.weight 
        bias = layer.bias 

        if verbose: print('layer: ', layer)
        if isinstance(layer, torch.nn.Conv2d):
            print('conv layer')
            in_shape = self._hooks[lk].in_shape
            
            # Apply padding
            stride = layer.stride 
            dilation = layer.dilation
            padding = layer.padding

            W_ = c2s(in_shape, weight, bias, stride=stride, padding=padding, dilation=dilation) 
            U, s, V = torch.svd_lowrank(W_, q=300)
            Vh = V.T

        elif isinstance(layer, torch.nn.Linear):
            if verbose: print('linear layer')
            W_ = torch.hstack((weight, bias.reshape(-1,1)))
            U, s, Vh = torch.linalg.svd(W_, full_matrices=False)
        else:
            raise RuntimeError('Unsuported layer type')

        _svds[lk] = TensorDict({
                'U': MMT(U.detach().cpu()),
                's': MMT(s.detach().cpu()),
                'Vh': MMT(Vh.detach().cpu())
                })

    if verbose: print(f'saving {file_path}')
    if len(_layers_to_compute) != 0:
        _svds.memmap(file_path)
    
    self._svds = _svds
    return self._svds
