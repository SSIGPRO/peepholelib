import sys
sys.path.insert(0, '/home/leandro/repos/peepholelib')

from tqdm import tqdm

import torch
from time import time
from numpy.random import randint as ri
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.functional import pad

from math import *


def channel_conv(layer):
    uw = layer.weight.flatten(start_dim=1, end_dim=-1)

    if not layer.bias == None:
        uw = torch.hstack([uw, layer.bias.view(-1,1)])

    return uw

def unroll_image(img, layer):
    if not isinstance(layer, torch.nn.Conv2d):
        raise RuntimeError('Input layer should be a torch.nn.Conv2D one')

    weight = layer.weight        
    bias = layer.bias
    groups = layer.groups
    pad_mode = layer.padding_mode
    
    # kernel offsets
    cic, coc = weight.shape[0], weight.shape[1]
    kh, kw = weight.shape[2], weight.shape[3]
    ph, pw = layer.padding
    sh, sw = layer.stride
    dh, dw = layer.dilation
    
    oh = int(floor((ih+2*ph - dh*(kh - 1) -1)/sh + 1))
    ow = int(floor((iw+2*pw - dw*(kw - 1) -1)/sw + 1))

    ui = torch.nn.functional.unfold(
            img,
            kernel_size = (kh, kw),
            dilation = (dh, dw),
            padding = (ph, pw),
            stride = (sh, sw)
            )

    if not layer.bias == None:
        ones = torch.ones(coc, 1, oh*ow).to(ui.device)
        ui = torch.hstack((ui, ones))

    return ui, oh, ow 


if __name__ == '__main__':
    torch.set_printoptions(linewidth=240, precision=2)
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device('cuda:4')#'f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    q = 300
    errors = []
    times = []
    torch.manual_seed(2)
    for i in range(1):
        using_bias = True 
        channel_wise = True 
        groups = 1#ri(1, 3)
        nc = groups*2#ri(2, 5) # multiple of groups
        kw = 2#ri(1, 2) # kernel width 
        kh = 2#ri(1, 2) # kernel height
        iw = 3#ri(2, 5) # image width
        ih = 3#ri(2, 5) # image height
        ns = 2 # n samples
        cic = nc # conv in channels 
        coc = groups*3#ri(2, 5) # conv out channels (multiple of groups)
        sh = 1#ri(2, 10)
        sw = 1#ri(2, 10)
        ph = 0#ri(2, 10) 
        pw = 0#ri(2, 10) 
        dh = 1#ri(2, 10) 
        dw = 1#ri(2, 10) 

        print('\n-------------------------')
        print('cic, coc: ', cic, coc)
        print('kernel h, w: ', kh, kw)
        print('image h, w: ', ih, iw)
        print('stride h, w: ', sh, sw)
        print('padding h, w: ', ph, pw)
        print('groups: ', groups)

        c = torch.nn.Conv2d(cic, coc, (kh, kw), bias=using_bias, stride=(sh, sw), dilation=(dh,dw), groups=groups, padding=(ph,pw), device=device)
        
        x = torch.rand(ns, nc, ih, iw).to(device)
        r = c(x).to(device)

        # get the sparse representation
        my_csr = channel_conv(c)
        print('unrolled kernel: ', my_csr)
        
        #print('img:\n', x)
        u_x_pad, oh, ow = unroll_image(x, c)
        #print('uxpad:\n', u_x_pad, u_x_pad.shape, ih, iw, ph, pw)
        #print('csr shape: ', my_csr.shape)

        #print('r:\n', r)
        rr = (my_csr@u_x_pad).reshape(ns, coc, oh, ow)
        print('erroe:', (rr-r).sum())

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
