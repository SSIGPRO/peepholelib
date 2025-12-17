import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

from tqdm import tqdm

import torch
from peepholelib.models.svd_fns import c2s
from time import time
from numpy.random import randint as ri
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.functional import pad

if __name__ == '__main__':
    torch.set_printoptions(linewidth=240)
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    q = 300
    errors = []
    times = []
    for i in range(30):
        using_bias = True 
        channel_wise = True 
        groups = ri(1, 3)
        nc = ri(20, 50)*groups # multiple of groups
        kw = ri(10, 20) # kernel width 
        kh = ri(10, 20) # kernel height
        iw = ri(20, 50) # image width
        ih = ri(20, 50) # image height
        ns = 1 # n samples
        cic = nc # conv in channels 
        coc = ri(20, 50)*groups # conv out channels (multiple of groups)
        sh = ri(2, 10)
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

        # pad input image
        pad_mode = c.padding_mode if c.padding_mode != 'zeros' else 'constant'
        x_pad = pad(x, pad=_reverse_repeat_tuple(c.padding, 2), mode=pad_mode) 
         
        # flatten and append 1 to the end of img if there is bias
        if using_bias:
            xu = torch.hstack((x_pad.flatten(), torch.ones(1))).to(device)
        else:
            xu = x_pad.flatten().to(device)

        # get the sparse representation
        my_csr = c2s(x[0].shape, c, channel_wise=channel_wise, device=device)

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
