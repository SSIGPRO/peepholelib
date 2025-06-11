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
    n = 5 
    bs = 2
    ds = 3
    device = 'cpu' 
    n_dicts = 2 
    
    # seems like batch size is really important
    td = PTD(filename='./banana', batch_size=[n], mode='w')
    for j in range(n_dicts):
        td['a%d'%j] = MMT.empty(shape=(n,ds))
        td['a%d'%j] = torch.rand((n, ds))
    
    p(td, 5)

    m = td.mean(dim=0)
    s = td.std(dim=0)
    for k in m.keys():
        print('mean: ', m[k])
        print('std: ', s[k])
    
    print('\n------------------------\n') 
    bs = 2
    dl = DataLoader(dataset=td, batch_size=bs, collate_fn=lambda x: x)
    for data in dl:
        for k in m.keys():
            print(f'{k} - before: ', data[k])
            data[k] = (data[k]-m[k])/s[k]
            print(f'{k} after: ', data[k])
            
    print('\nafter norm')
    p(td, 5)

    del td
    print('after loading')
    td2 = PTD.from_h5('./banana', mode='r')
    p(td2, 5)
    print('--------------')
    print(td2)
    td2.del_('a0')
    print(td2)
    print('--------------')

    del td2

    td3 = PTD.from_h5('./banana', mode='r')
    print(td3)
