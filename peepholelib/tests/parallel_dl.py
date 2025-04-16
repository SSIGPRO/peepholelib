from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT

from pathlib import Path
import torch
import sys
from torch.utils.data import DataLoader
import time

def p(td, bs, nw=1):
    if nw == 1:
        dl = DataLoader(td, batch_size=bs, collate_fn=lambda x:x)
    else:
        dl = DataLoader(td, batch_size=bs, collate_fn=lambda x:x, num_workers=nw)
    for d in dl:
        for k in td.keys():
            print(f'td[{k}] = ', d[k])

if __name__ == '__main__':
    n = 5 
    bs = 2
    ds = 3
    device = torch.device('cuda:4') 
    n_dicts = 1 
    file = Path('banana') 
    
    print('Creating')
    td = PTD(filename=file, mode='w', batch_size=[n], device=device)
    for j in range(n_dicts):
        td['a%d'%j] = MMT.empty(shape=(n,ds))
    p(td, n) 
    td.close()
    
    td = PTD.from_h5(filename=file, mode='r+')
    print('Filling')
    for j in range(n_dicts+1):
        td['a%d'%j] = torch.rand((n, ds))
    p(td, n, nw=2) 
    td.close()

    print('Printing')
    td = PTD.from_h5(filename=file, mode='r')
    print(td.is_consolidated(), td.is_contiguous(), td.is_memmap(), td.is_shared())
    p(td, bs, nw=2) 

