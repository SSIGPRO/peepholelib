from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT

import torch
from torch.utils.data import DataLoader
import h5py

from pathlib import Path
from time import time

####
# what we learn
# 1. chuncks seem to help, but does not work with data of different shapes
# 2. auto chuncking works, but we have implications on the time
# 3. external divides into small files, but sort of buggy depending on the size (cs)

if __name__ == '__main__':
    ns = 2**15
    ds = 2**8
    cs = 2**13
    bs = 2**5
    it = 10
    fn = Path('./data/banana')
    device = torch.device('cuda:0')
    
    if fn.exists():
        td = PTD.from_h5(filename=fn, mode='r')
        print('loading')
    else:
        print('n samples: ', ns)
        nc = ns//cs
        external = [(f'data/{i}', i*cs, cs) for i in range(nc)]
        if ns-cs*nc > 0:
            external += [(f'data/{nc}', cs*nc, ns-cs*nc)]
        print(external, cs*nc, ns-cs*nc)

        td = PTD(filename=fn, batch_size=[ns], mode='w', external=external) 
        td['a'] = MMT.empty(shape=(ns, ds)) 
        td['b'] = MMT.empty(shape=(ns, 1)) 
        
        dl = DataLoader(td, batch_size=bs, collate_fn=lambda x: x)
        for d in dl:
            d['a'] = torch.rand(d.shape[0], ds)
            d['b'] = torch.rand(d.shape[0], 1)

    print('computing')

    ts = []
    idx = torch.randperm(ns) 
    x = torch.zeros(ds, device=device)
    for i in range(it):
        t0 = time()
        x += td[idx[i]]['a'].to(device)/it
        t = time()-t0
        ts.append(t)
    print('mean time: ', torch.tensor(ts).mean())


