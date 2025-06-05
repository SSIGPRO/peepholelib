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
    weight = layer.weight        
    bias = layer.bias
    #print('weights: ', weight)
    uw = weight.flatten(start_dim=1, end_dim=-1)
    #print('unrolled weights: ', uw)

    return uw

def unroll_image(img, layer):
    if not isinstance(layer, torch.nn.Conv2d):
        raise RuntimeError('Input layer should be a torch.nn.Conv2D one')

    weight = layer.weight        
    bias = layer.bias
    groups = layer.groups
    stride = layer.stride
    padding = layer.padding
    dilation = layer.dilation
    pad_mode = layer.padding_mode

    if dilation != (1,1):
            raise RuntimeError('This functions does not account for dilation, if you extendent it, please send us a PR ;).')

    ns = img.shape[0]
    nc = img.shape[1]
    ih = img.shape[2]
    iw = img.shape[3]
    
    nco, nci = weight.shape[0], weight.shape[1]

    # kernel offsets
    kh, kw = weight.shape[2], weight.shape[3]
    ph, pw = padding
    sh, sw = stride
    dh, dw = dilation

    oh = int(floor((ih - dh*(kh - 1) -1)/sh + 1))
    ow = int(floor((iw - dw*(kw - 1) -1)/sw + 1))
    print('out shape: ', oh, ow)

    koht = floor(kh/2) 
    kohb = kh - floor(kh/2) - 1 
    kowr = floor(kw/2) 
    kowl = kw - floor(kw/2) - 1
    print('kernel offsets: ', koht, kohb, kowr, kowl)

    assert nc == nci, f'Number of channels in the input image {nc} does not match with the conv layer {nci}'
    
    # pad image
    pad_mode = pad_mode if pad_mode != 'zeros' else 'constant'
    ip = pad(img, pad=_reverse_repeat_tuple((ph, pw), 2), mode=pad_mode) 
    print('img pad:\n', ip)
    
    ui = torch.zeros(ns, nci*kh*kw, oh*ow)
    for _ho, _h in enumerate(range(koht, ih-kohb)):
        for _wo, _w in enumerate(range(kowl, iw-kowr)):
            print('limits: ', max(_h-koht, 0), min(_h+kohb+1, ih), max(_w-kowl,0), min(_w+kowr+1, iw))
            oip = ip[:,:, max(_h-koht, 0):min(_h+kohb+1, ih), max(_w-kowl,0):min(_w+kowr+1, iw)]
            print('oip: ', oip)
            oipf = oip.flatten(start_dim=1, end_dim=-1)
            print('oipf: ', oipf)
            ui[:,:, _ho*ow+_wo] = oipf
    print(ui)

    '''
    # flattens the padded image
    ipf = ip.reshape(ns, nc, (ih+2*ph)*(iw+2*pw))
    print('ipf:\n', ipf)

    # repeats it for each kernel dimension
    ipfr = ipf.repeat(1, 1, kh*kw).reshape(ns, nc, kh*kw, (ih+2*ph)*(iw+2*pw))

    # rotates the image repetitions to align then wirh the kernel values
    base_shift = kohb*(iw+2*pw)+kowl 
    print('base shift: ', base_shift)

    for _h in range(kh):
        for _w in range(kw):
            ipfr[:,:, _w+_h*kw] = ipfr[:,:, _w+_h*kw].roll(base_shift-(_w+_h*(iw+2*pw)))
    '''
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
    for i in range(1):
        using_bias = False 
        channel_wise = True 
        groups = 1#ri(1, 3)
        nc = 1#ri(2, 5)*groups # multiple of groups
        kw = 2#ri(1, 2) # kernel width 
        kh = 2#ri(1, 2) # kernel height
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
        my_csr = channel_conv(c)
        print('unrolled kernel: ', my_csr)
        
        print('img:\n', x)
        u_x_pad, oh, ow = unroll_image(x, c)
        print('uxpad:\n', u_x_pad, u_x_pad.shape, ih, iw, ph, pw)
        print('csr shape: ', my_csr.shape)

        print('r:\n', r)
        rr = (my_csr@u_x_pad).reshape(ns, coc, oh, ow)
        print('rr:\n', rr)

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
