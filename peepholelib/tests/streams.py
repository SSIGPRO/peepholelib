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
# 4. No real difference using streams, synching is making it run sequentially

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    dev_cp = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    dev_sv = torch.device('cpu')
    print(f"Using {dev_cp} for computing and {dev_sv} for saving")

    ns = 2**3#19
    ds = 2**2#13
    bs = 2**2#17
    n_threads = 4
    src = Path('./aaa')
    dst = Path('./bbb')
    dst0 = Path('./ccc')
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
    
    # dest PTD reference 
    if not dst0.exists():
        td_dst0 = PTD(filename=dst0, batch_size=[ns], mode='w')
        for k in ks:
            td_dst0[k] = MMT.empty(shape=(ns, ds))
    else:
        td_dst0 = PTD.from_h5(filename=dst0, mode = 'r+')

    # copying data
    dl_src = DL(td_src, batch_size=bs, collate_fn=lambda x:x)
    dl_dst = DL(td_dst, batch_size=bs, collate_fn=lambda x:x)
    
    # streams for sending, computing and receiving data
    s_snd = torch.cuda.Stream(device=dev_cp)
    s_cmp = torch.cuda.Stream(device=dev_cp)
    _d = None

    t0 = time()
    for d_s, d_t in zip(dl_src, dl_dst):
        for k in ks:
            with torch.cuda.stream(s_snd):
                _d = d_s[k].to(dev_cp, non_blocking=True)
                _d.record_stream(s_cmp)
    
            if _d != None:
                with torch.cuda.stream(s_cmp):
                    s_cmp.wait_stream(s_snd)
                    _dp = _d/2
                    d_t[k] = _dp 

    print('new time: ', time()-t0) 
    
    # compute ref
    dl_src = DL(td_src, batch_size=bs, collate_fn=lambda x:x)
    dl_dst0 = DL(td_dst0, batch_size=bs, collate_fn=lambda x:x)
    t0 = time()
    for d_s, d_t0 in zip(dl_src, dl_dst0):
        for k in ks:
            d_t0[k] = d_s[k].to(dev_cp)/2
 
    print('old time: ', time()-t0) 

    # check correctness 
    td_dst.close()
    td_dst0.close()
    td_dst = PTD.from_h5(filename=dst, mode = 'r+')
    td_dst0 = PTD.from_h5(filename=dst0, mode = 'r+')
    dl_dst = DL(td_dst, batch_size=bs, collate_fn=lambda x:x)
    dl_dst0 = DL(td_dst0, batch_size=bs, collate_fn=lambda x:x)
    for d_t, d_t0 in zip(dl_dst, dl_dst0):
        corr = True
        for k in ks:
            corr = corr and (d_t[k] == d_t0[k]).all()
        print(f'batch correct: {corr}')

    # close and re-open
    td_src.close()
    td_dst.close()
    td_dst0.close()
    quit()
