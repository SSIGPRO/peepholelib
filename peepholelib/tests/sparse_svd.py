import sys
sys.path.insert(0, '/home/leandro/repos/peepholelib')

import random
import torch
from peepholelib.models.svd import c2s as c2s
from time import time
from numpy.random import randint as ri
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.functional import pad

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    using_bias = True 

    for i in range(30):
        groups = ri(1, 3)
        nc = ri(1, 5)*groups # multiple of groups
        kw = ri(2, 10) # kernel width 
        kh = ri(2, 10) # kernel height
        iw = ri(10, 50) # image width
        ih = ri(10, 50) # image height
        ns = 1 # n samples
        cic = nc # conv in channels 
        coc = ri(2, 20)*groups # conv out channels (multiple of groups)
        sh = ri(2 ,10)
        sw = ri(2, 10)
        ph = ri(2, 10) 
        pw = ri(2, 10) 

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

        t0 = time()
        # pad input image
        pad_mode = c.padding_mode if c.padding_mode != 'zeros' else 'constant'
        x_pad = pad(x, pad=_reverse_repeat_tuple(c.padding, 2), mode=pad_mode) 
        
        my_csr = c2s(x[0].shape, c, device=device)

        #'''
        print('SVDing')
        s, v, d = torch.svd_lowrank(my_csr, q=300)
        #print('v:',  v)
        print('SVDone')
        #'''
        t_curr = time()-t0

        lc = my_csr.to_dense()
        if using_bias:
            xu = torch.hstack((x_pad.flatten(), torch.ones(1))).to(device)
        else:
            xu = x_pad.flatten().to(device)
        # peephole
        ru = lc@xu
        error = torch.norm(r-ru.reshape(r.shape))/torch.norm(r) 
        print('error ru: ', error)
        print('time: ', t_curr) 
        if error > 1.0:
            raise RuntimeError('Girl, go debug that conv.')
        print('-------------------------\n')
