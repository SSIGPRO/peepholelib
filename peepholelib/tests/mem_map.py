from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
import torch
import sys
from torch.utils.data import DataLoader

def p(td, bs):
    dl = DataLoader(dataset=td, batch_size=bs, collate_fn=lambda x: x)
    print('\ndataloader: ', len(dl.dataset))
    for d in dl:
        print('batch')
        for kk in d.keys():
            print(kk, d[kk].contiguous())
            if kk == 'b':
                for k in d['b'].keys():
                    print('b', k, d['b'][k].contiguous())

if __name__ == '__main__':
    n = 5 
    bs = 2
    ds = 3
    device = 'cpu' 
    n_dicts = 2 
    
    # seems like batch size is really important
    td = TensorDict(batch_size=n, device=device)
    for j in range(n_dicts):
        td['a%d'%j] = MMT.empty(shape=(n,ds))
    
    td = td.memmap_like('./banana')
    # makes one dict for each input
    for i in range(n):
        d = {}
        # create data
        for j in range(2):
            d['a%d'%j] = torch.rand((1, ds))
        td[i] = TensorDict(d) 
    print('saving', sys.getsizeof(td))

    # try add stuff after memmap
    del td
    td = TensorDict.load_memmap('./banana')
    td['b'] = TensorDict(batch_size=n)
    for j in range(n_dicts):
        td['b']['b%d'%j] = MMT.empty(shape=(n,ds))
    td = td.memmap_like('./banana')
    
    for i in range(n):
        d = {'b':{}}
        # create data
        for j in range(2):
            d['b']['b%d'%j] = torch.rand((1, ds))
        td[i] = TensorDict(d) 
    # must memmap again
    print('saving again')
    p(td, 5)
    del td

    td2 = TensorDict.load_memmap('./banana')
    print('\n\nloading td2:', td2)
    p(td2, 3)
