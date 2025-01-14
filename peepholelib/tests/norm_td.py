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
    
    p(td, 5)

    m = td.mean(dim=0)
    s = td.std(dim=0)
    print('mean: ', m)
    print('std: ', s)
    print('\n------------------------\n') 
    bs = 2
    dl = DataLoader(dataset=td, batch_size=bs, collate_fn=lambda x: x)
    for bn, data in enumerate(dl):
        n_in = len(data)
        temp = (data-m)/s
        td[bn*bs:bn*bs+n_in] = temp 

    print('after norm')
    p(td, 5)

    del td
    print('after loading')
    td2 = TensorDict.load_memmap('./banana')
    p(td2, 5)
