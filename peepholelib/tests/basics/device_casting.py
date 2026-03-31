from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT

import torch
from torch.utils.data import DataLoader as DL
from cuda_selector import auto_cuda

from pathlib import Path
from time import time

#-----------------------------
# things we learned
#
# 1. destine PTD should always be on cpu
# 2. better to send individual PTD keys to GPU and not set the PTD device
# 3. pin_memory on dataloader seems to be sligtly faster, but inconsistent. Not sure it is a good idea

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    dev_cp = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    dev_sv = torch.device('cpu')
    print(f"Using {dev_cp} for computing and {dev_sv} for saving")

    ns = 2**19
    ds = 2**13
    bs = 2**17
    n_threads = 4
    src = Path('./aaa')
    dst = Path('./bbb')
    ks = ['a', 'b', 'c']
    
    #  create src PTD
    if not src.exists():
        td_src = PTD(filename=src, batch_size=[ns], mode='w')
        for k in ks:
            td_src[k] = MMT.empty(shape=(ns, ds))
        
        dl = DL(td_src, batch_size=bs, collate_fn=lambda x:x)
        for d in dl:
            s = len(d)
            for k in d.keys():
                d[k] = torch.rand(s, ds)

    else:
        td_src = PTD.from_h5(filename=src, mode = 'r+')

    # dest PTD 
    if not dst.exists():
        td_dst = PTD(filename=dst, batch_size=[ns], mode='w')
        for k in ks:
            td_dst[k] = MMT.empty(shape=(ns, ds))
    else:
        td_dst = PTD.from_h5(filename=dst, mode = 'r+')

    # copying data
    dl_src = DL(td_src, batch_size=bs, collate_fn=lambda x:x)
    dl_dst = DL(td_dst, batch_size=bs, collate_fn=lambda x:x)

    for d_s, d_t in zip(dl_src, dl_dst):
        t0 = time()
        for k in ks:
            _d = d_s[k].to(dev_cp)
            _dp = _d/2
            d_t[k] = _dp 
        print('time: ', time()-t0) 
    
    # close and re-open
    td_src.close()
    td_dst.close()
    quit()

    td = PTD.from_h5(filename=dst, mode = 'r')
    
    # print to check device
    dl = DL(td, batch_size=bs, collate_fn=lambda x:x)
    for d in dl:
        for k in d.keys():
            print(f'{k}: {d[k]}')
