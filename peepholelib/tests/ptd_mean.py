from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT
import torch
import sys
from torch.utils.data import DataLoader

def p(td, bs):
    dl = DataLoader(dataset=td, batch_size=bs, collate_fn=lambda x: x)
    print('\n------: ')
    for data in dl:
        for kk in data.keys():
            print(kk, data[kk].contiguous())

if __name__ == '__main__':
    n = 3 
    bs = 2
    ds = (3, 2)
    device = 'cpu' 
    n_dicts = 2 
    
    # seems like batch size is really important
    td = PTD(filename='./banana', batch_size=[n], mode='w')
    for j in range(n_dicts):
        td['a%d'%j] = MMT.empty(shape=(n,)+ds)
        td['a%d'%j] = torch.rand((n,)+ds)
    
    td2 = PTD(filename='./banana2', batch_size=[n], mode='w')
    td2['l'] = MMT.empty(shape=(n,))
    td2['l'] = torch.randint(0, 3, (n,))
    
    p(td, n)
    p(td2, n)
    
    for i in [0, 1, 2]:
        idx = td2['l'] == i
        print(idx)
        _t = td[idx]
        p(_t, len(_t))

    m = td.mean(dim=0)
    s = td.std(dim=0)
    for k in m.keys():
        print('mean: ', m[k])
        print('std: ', s[k])

    print('a: \n', td['a0'])
    print('a0_a1: \n', (td['a0'][0]+td['a0'][1]+td['a0'][2])/3)
    print('a m: \n', td['a0'].mean(dim=0))

