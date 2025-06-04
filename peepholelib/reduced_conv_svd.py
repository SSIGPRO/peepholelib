import sys
sys.path.insert(0, '/home/leandro/repos/peepholelib')

from tqdm import tqdm

import torch
from time import time
from numpy.random import randint as ri
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.functional import pad

from math import *


def channel_conv(input_shape, layer, device='cpu', verbose=False, warns=True):
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
        #if warns: warn('Do not forget to pad your input accoding to the Conv2d padding. Deactivate this warning passing warns=False as argument.', stacklevel=2)
    
    Cin, Hin, Win = input_shape
    Cout = weight.shape[0]
    Hk = weight.shape[2]
    Wk = weight.shape[3]
                                                                                  
    #print('weights: ', weight)
    uw = weight.flatten(start_dim=1, end_dim=-1)
    #print('unrolled weights: ', uw)

    return uw

def unroll_image(img, kh, kw, ph, pw, pad_mode):
    ns = img.shape[0]
    nc = img.shape[1]
    ih = img.shape[2]
    iw = img.shape[3]

    if pad_mode == 'zeros':
        pad_mode = 'constant'
    ip = pad(img, pad=_reverse_repeat_tuple((ph, pw), 2), mode=pad_mode) 
    #print('img pad:\n', ip)

    # flattens the padded image
    ipf = ip.reshape(ns, nc, (ih+2*ph)*(iw+2*pw))
    print('ipf:\n', ipf)

    # repeats it for each kernel dimension
    ipfr = ipf.repeat(1, 1, kh*kw).reshape(ns, nc, kh*kw, (ih+2*ph)*(iw+2*pw))
    # rotates the image repetitions to align then wirh the kernel values
    #base_shift = iw+2*pw+1
    #print('base shift: ', base_shift)
    for _h in range(kh):
        for _w in range(kw):
            print(2*kw*_w+2*kh*_h)
            ipfr[:,:, _w+_h*kw] = ipfr[:,:, _w+_h*kw].roll(-(_w+_h*(iw+2*pw)))

    return ipfr


if __name__ == '__main__':
    torch.set_printoptions(linewidth=240, precision=2)
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device('cuda:4')#'f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    q = 300
    errors = []
    times = []
    for i in range(1):
        using_bias = False 
        channel_wise = True 
        groups = 1#ri(1, 3)
        nc = 1#ri(2, 5)*groups # multiple of groups
        kw = 3#ri(1, 2) # kernel width 
        kh = 3#ri(1, 2) # kernel height
        iw = 3#ri(2, 5) # image width
        ih = 3#ri(2, 5) # image height
        ns = 1 # n samples
        cic = nc # conv in channels 
        coc = 1#ri(2, 5)*groups # conv out channels (multiple of groups)
        sh = 1#ri(2, 10)
        sw = 1#ri(2, 10)
        ph = 1#ri(2, 10) 
        pw = 1#ri(2, 10) 

        print('\n-------------------------')
        print('cic, coc: ', cic, coc)
        print('kernel h, w: ', kh, kw)
        print('image h, w: ', ih, iw)
        print('stride h, w: ', sh, sw)
        print('padding h, w: ', ph, pw)
        print('groups: ', groups)

        c = torch.nn.Conv2d(cic, coc, (kh, kw), bias=using_bias, stride=(sh, sw), dilation=(1,1), groups=groups, padding=(ph,pw))
        
        x = torch.rand(ns, nc, ih, iw)
        r = c(x).to(device)

        # pad input image
        pad_mode = c.padding_mode if c.padding_mode != 'zeros' else 'constant'
        x_pad = pad(x, pad=_reverse_repeat_tuple(c.padding, 2), mode=pad_mode) 
         
        # flatten and append 1 to the end of img if there is bias
        if using_bias:
            xu = torch.hstack((x_pad.flatten(), torch.ones(1))).to(device)
        else:
            xu = x_pad.flatten().to(device)

        # get the sparse representation
        my_csr = channel_conv(x[0].shape, c, device=device)
        print('unrolled kernel: ', my_csr)
        
        print('img:\n', x)
        u_x_pad = unroll_image(x, kh, kw, ph, pw, pad_mode=c.padding_mode)
        print('uxpad:\n', u_x_pad, u_x_pad.shape, ih, iw, ph, pw)
        print('csr shape: ', my_csr.shape)

        rr = (my_csr@u_x_pad).reshape(coc, ih+2*ph, iw+2*pw)
        print('rr:\n', rr)
        print('r:\n', r)

        quit()
        
        t0 = time()
        print('SVDing')
        if isinstance(my_csr, list):
            lc = []
            for c in tqdm(my_csr):
                _s, _v, _d = torch.svd_lowrank(c, q=q)
                _lc = (_s@torch.diag(_v)@_d.T)

                # move partial results to CPU to not fill the GPU
                lc.append(_lc.detach().cpu())
            lc = torch.vstack(lc).to(device)
        else:
            s, v, d = torch.svd_lowrank(my_csr, q=q)
            lc = s@torch.diag(v)@d.T 

        print('SVDone')
        print()
        t_curr = time()-t0

        ru = lc@xu
        error = torch.norm(r-ru.reshape(r.shape))/torch.norm(r) 
        print('error ru: ', error)
        print('time: ', t_curr) 
        errors.append(error)
        times.append(t_curr)
        if error > 0.1:
            raise RuntimeError('Girl, go debug that conv.')
        print('-------------------------\n')
    print('mean time: ', torch.tensor(times).mean())
    print('mean errs: ', torch.hstack(errors).mean())
