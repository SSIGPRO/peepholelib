import numpy as np
from warnings import warn
from pathlib import Path

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

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

    #if verbose:
    #    print(f"cin: {Cin}, cout: {Cout}, groups: {groups}")
    #    print(" bias:", bias.shape if bias is not None else None)
    #    print(f"nº of Cin per group: {Cin_g}, nº of Cout per group: {Cout_g}")
    #    print(f"kernel height: {Hk}, kernel width: {Wk}, kernel size: {kernel_size}")
    #    print(f"input height: {Hin}, input width: {Win}, input size: {input_size}")

    Hout = int(np.floor((Hin - dilation[0]*(Hk - 1) -1)/stride[0] + 1))
    Wout = int(np.floor((Win - dilation[1]*(Wk - 1) -1)/stride[1] + 1))
    output_size = Hout*Wout

    shape_out = torch.Size((Cout*output_size, Cin*input_size + (0 if bias is None else 1)))
    
    crow = (torch.linspace(0, shape_out[0], shape_out[0]+1)*(kernel_size*Cin + (0 if bias is None else 1))).int()
    nnz = crow[-1]
    
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
        #if verbose: print(f" Group {group_id}: indexes in: {start_index_in}-{end_index_in} - indexes out: {start_index_out}-{end_index_out}")

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

    cols = cols.flatten()
    data = data.flatten()

    csr_mat = torch.sparse_csr_tensor(crow, cols, data, size=shape_out, device=device)
    return csr_mat 

def get_svds(self, **kwargs):
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    device = kwargs['device'] if 'device' in kwargs else 'cpu'
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
            in_shape = self._hooks[lk].in_shape
            
            W_ = c2s(in_shape, layer, device=device) 
            U, s, V = torch.svd_lowrank(W_, q=300)
            Vh = V.T

        elif isinstance(layer, torch.nn.Linear):
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
