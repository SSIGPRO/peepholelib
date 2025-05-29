# General python stuff
from warnings import warn

# torch stuff
import torch
from math import *

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
    if Cout % groups != 0: raise RuntimeError("Cout must be divisible by groups")
    
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

def ct2s(input_shape, layer, channel_wise=False, device='cpu', verbose=False, warns=True):
    if not isinstance(layer, torch.nn.ConvTranspose2d):
        raise RuntimeError('Input layer should be a torch.nn.Conv2D or torch.nn.ConvTranspose2d')

    weight = layer.weight
    bias = layer.bias
    groups = layer.groups
    stride = layer.stride
    padding = layer.padding
    dilation = layer.dilation
    output_padding = layer.output_padding

    if padding != (0,0) or dilation != (1,1) or groups!=1 or channel_wise == True:
        raise RuntimeError('This functions does not account for padding, dilation, groups, and channelwise if you extendent it, please send us a PR ;).')
    
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
