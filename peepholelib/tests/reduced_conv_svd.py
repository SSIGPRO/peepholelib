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
    _ih, _iw = img.shape[2], img.shape[3] 
    kh, kw = weight.shape[2], weight.shape[3]
    ph, pw = layer.padding
    sh, sw = layer.stride
    dh, dw = layer.dilation
    
    oh = int(floor((_ih+2*ph - dh*(kh - 1) -1)/sh + 1))
    ow = int(floor((_iw+2*pw - dw*(kw - 1) -1)/sw + 1))

    ui = torch.nn.functional.unfold(
            img,
            kernel_size = (kh, kw),
            dilation = (dh, dw),
            padding = (ph, pw),
            stride = (sh, sw)
            )

    if not layer.bias == None:
        ones = torch.ones(ui.shape[0], 1, oh*ow).to(ui.device)
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
    for i in range(30):
        banana = 'potato'
        using_bias = True 
        channel_wise = True 
        groups = 1#ri(1, 3)
        nc = groups*ri(20, 50) # multiple of groups
        kw = ri(10, 20) # kernel width 
        kh = ri(10, 20) # kernel height
        iw = ri(40, 50) # image width
        ih = ri(40, 50) # image height
        ns = 1 # n samples
        cic = nc # conv in channels 
        coc = groups*ri(20, 50) # conv out channels (multiple of groups)
        sh = ri(2, 10)
        sw = ri(2, 10)
        ph = ri(2, 10) 
        pw = ri(2, 10) 
        dh = ri(1, 3) 
        dw = ri(1, 3) 

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
        ux, oh, ow = unroll_image(x, c)
        
        t0 = time()
        print('SVDing')
        s, v, d = torch.linalg.svd(my_csr, full_matrices=False)
        print('SVDone')
        t_curr = time()-t0

        lc = s@torch.diag(v)@d 
        ru = (lc@ux).reshape(ns, coc, oh, ow)
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
