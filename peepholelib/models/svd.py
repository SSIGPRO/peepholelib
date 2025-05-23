# General python stuff
from warnings import warn
from pathlib import Path
import numpy as np
from tqdm import tqdm

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

def c2s(input_shape, layer, channel_wise=False, device='cpu', verbose=False, warns=True):
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

    Hout = int(np.floor((Hin - dilation[0]*(Hk - 1) -1)/stride[0] + 1))
    Wout = int(np.floor((Win - dilation[1]*(Wk - 1) -1)/stride[1] + 1))
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

    if not channel_wise:
        shape_out = torch.Size((Cout*output_size, Cin*input_size + (0 if bias is None else 1)))
        crow = (torch.linspace(0, shape_out[0], shape_out[0]+1)*(kernel_size*Cin + (0 if bias is None else 1))).int()

        cols = cols.flatten()
        data = data.flatten()
        
        csr_mat = torch.sparse_csr_tensor(crow, cols, data, size=shape_out, device=device)

        ret = csr_mat
    else:
        csrs = []
        shape_out = torch.Size((output_size, Cin*input_size + (0 if bias is None else 1)))
        crow = (torch.linspace(0, shape_out[0], shape_out[0]+1)*(kernel_size*Cin + (0 if bias is None else 1))).int()

        for cout in range(Cout):
            hl, hh = cout*output_size, (cout+1)*output_size
            _cols = cols[hl:hh, :] 
            _data = data[hl:hh, :] 
            _cols = _cols.flatten()
            _data = _data.flatten()

            csrs.append(torch.sparse_csr_tensor(crow, _cols, _data, size=shape_out, device=device))

        ret = csrs
    return ret 


def get_svds(self, **kwargs):
    path = Path(kwargs['path'])
    name = kwargs['name']
    target_modules = kwargs['target_modules'] 
    sample_in = kwargs['sample_in']
    q = kwargs['rank'] if 'rank' in kwargs else 300
    channel_wise = kwargs['channel_wise'] if 'channel_wise' in kwargs else True
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    # create folder
    path.mkdir(parents=True, exist_ok=True)
    
    file_path = path/name
    if file_path.exists():
        if verbose: print(f'File {file_path} exists. Loading from disk.')
        _svds = TensorDict.load_memmap(file_path)
    else: 
        _svds = TensorDict()

    _modules_to_compute = []
    for mk in target_modules:
        if mk in _svds.keys():
            continue
        _modules_to_compute.append(mk)
    if verbose: print('modules to compute SVDs: ', _modules_to_compute)
    
    for mk in _modules_to_compute:
        if verbose: print(f'\n ---- Getting SVDs for {mk}\n')
        module = self._target_modules[mk]
        weight = module.weight 
        bias = module.bias 

        if isinstance(module, torch.nn.Conv2d):
            # dry run to get shape
            self.set_activations(save_input=True, save_output=False)
            with torch.no_grad():
                _in = sample_in.reshape((1,)+sample_in.shape).to(self.device)
                self(_in)
                in_shape = self._acts['in_activations'][mk].shape[1:]
            self.set_activations(save_input=False, save_output=False)

            W_ = c2s(in_shape, module, channel_wise=channel_wise, device=self.device) 

            # same as `if channel_wise:`
            if isinstance(W_, list):
                uu, ss, vv = [], [], []
                for csr in tqdm(W_):
                    _u, _s, _v = torch.svd_lowrank(csr, q=q)
                    uu.append(_u.detach().cpu())
                    ss.append(_s.detach().cpu())
                    vv.append(_v.detach().cpu().T)
                U = torch.stack(uu)
                s = torch.stack(ss)
                Vh = torch.stack(vv)
            else:
                U, s, V = torch.svd_lowrank(W_, q=q)
                U, s, Vh = U.detach().cpu(), s.detach().cpu(), V.detach().cpu().T

        elif isinstance(module, torch.nn.Linear):
            W_ = torch.hstack((weight, bias.reshape(-1,1)))
            U, s, Vh = torch.linalg.svd(W_, full_matrices=False)
            U, s, Vh = U.detach().cpu(), s.detach().cpu(), Vh.detach().cpu()
        else:
            raise RuntimeError('Unsuported layer type')

        _svds[mk] = TensorDict({
                'U': MMT(U),
                's': MMT(s),
                'Vh': MMT(Vh)
                })

    if verbose: print(f'saving {file_path}')
    if len(_modules_to_compute) != 0:
        _svds.memmap(file_path)
    
    self._svds = _svds
    return self._svds
