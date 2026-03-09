import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

from tqdm import tqdm

import torch
from peepholelib.models.svd_fns import c2s
from warnings import warn
from time import time
from numpy.random import randint as ri
from math import floor
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.functional import pad

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
            data = torch.hstack((data, bias[cin]*torch.ones(Hin*Win))) 

    ret = torch.sparse_csr_tensor(crow, cols, data, size=shape_out, device=device)
    ret = ret.t()
    return ret 

if __name__ == '__main__':
    torch.set_printoptions(linewidth=240)
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:3" if use_cuda else "cpu")
    print(f"Using {device} device")
    q = 300
    errors = []
    times = []
    for i in range(50):
        using_bias = True 
        channel_wise = False 
        ms = 10 # master size
        groups = 1#ri(1, 3)
        nc = groups*ri(2*ms, 5*ms) # multiple of groups
        kw = ri(1*ms, 2*ms) # kernel width 
        kh = ri(1*ms, 2*ms) # kernel height
        iw = ri(4*ms, 5*ms) # image width
        ih = ri(4*ms, 5*ms) # image height
        ns = 1 # n samples
        cic = nc # conv in channels 
        coc = groups*ri(2*ms, 5*ms) # conv out channels (multiple of groups)
        sh = ri(2*ms, 3*ms)
        sw = ri(2*ms, 3*ms)
        ph = 0#ri(1*ms, 2*ms) 
        pw = 0#ri(1*ms, 2*ms) 
        oph = ri(1*ms, 2*ms)
        opw = ri(1*ms, 2*ms)
        dh = 0#ri(1*ms, 3*ms) 
        dw = 0#ri(1*ms, 3*ms) 

        print('\n-------------------------')
        print('cic, coc: ', cic, coc)
        print('kernel h, w: ', kh, kw)
        print('image h, w: ', ih, iw)
        print('stride h, w: ', sh, sw)
        print('padding h, w: ', ph, pw)
        print('dilation h, w: ', dh, dw)
        print('groups: ', groups)
        
        c = torch.nn.Conv2d(cic, coc, (kh, kw), bias=using_bias, stride=(sh, sw), dilation=(1,1), groups=groups, padding=(ph,pw))
        ct = torch.nn.ConvTranspose2d(coc, cic, (kh, kw), bias=using_bias, stride=(sh, sw), dilation=(1,1), groups=groups, padding=(ph,pw), output_padding=(oph, opw))
        ct.weight.requires_grad_(False)
        ct.weight[:] = c.weight[:]

        x = torch.rand(ns, nc, ih, iw)
        r = c(x)
        rt = ct(r)
        print('r shapes: ', r.shape, rt.shape)

        # pad input image
        pad_mode = c.padding_mode if c.padding_mode != 'zeros' else 'constant'
        x_pad = pad(x, pad=_reverse_repeat_tuple(c.padding, 2), mode=pad_mode) 
         
        # flatten and append 1 to the end of img if there is bias
        if using_bias:
            xu = torch.hstack((x_pad.flatten(), torch.ones(1)))
        else:
            xu = x_pad.flatten()

        # get the sparse representation
        csr = c2s(x[0].shape, c, channel_wise=False)
        csrt = ct2s(r[0].shape, ct, channel_wise=False)

        t0 = time()
        print('SVDing')
        s, v, d = torch.svd_lowrank(csr, q=q)
        lc = s@torch.diag(v)@d.T 
        print('SVDone. Time: ', time()-t0)
        ru = lc@xu

        t0 = time()
        print('SVDing')
        st, vt, dt = torch.svd_lowrank(csrt, q=q)
        lct = st@torch.diag(vt)@dt.T 
        print('SVDone. Time: ', time()-t0)
        xut = pad(r, pad=_reverse_repeat_tuple(c.padding, 2), mode=pad_mode).flatten() 
        if using_bias: xut = torch.hstack((xut, torch.ones(1)))
        rut = lct@xut

        error = torch.norm(r-ru.reshape(r.shape))/torch.norm(r) 
        print('ru shape: ', ru.shape, error)

        errort = torch.norm(rt-rut.reshape(rt.shape))/torch.norm(rt) 
        print('rut shape: ', rut.shape, errort)

        errors.append(error)
        if error > 0.1 or errort > 0.1:
            raise RuntimeError('Girl, go debug that conv.')
        print('-------------------------\n')
    print('mean errs: ', torch.hstack(errors).mean())
