import torch
from tensordict import TensorDict as TD
from torch.utils.data import DataLoader as DL
from torch.utils.data import random_split
from torch.utils.data import Dataset 
from torch.utils.data import Subset 

def foo(x):
    print('foo')
    return x

def bar(x):
    print('bar')
    return x

class DS(Dataset):
    def __init__(self, data, t=None):
        self.len = len(data['a'])
        self.data = data
        self.trans = t

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.trans:
            r = self.trans(self.data[idx])
        else:
            r = self.data[idx]
        return r 

    def __getitems__(self, idx):
        if self.trans:
            r = self.trans(self.data[idx])
        else:
            r = self.data[idx]
        return r 

if __name__ == '__main__':
    ns = 10
    td = TD({
        'a': torch.rand(ns, 2),
        'b': torch.rand(ns, 2)
        }, batch_size=ns)

    for k in td.keys():
        print(f'{k} - {td[k]}')

    idx_foo, idx_bar = random_split(
            range(ns),
            [0.5, 0.5],
            generator=torch.Generator().manual_seed(343)
            )

    d_foo = Subset(DS(td, t=foo), indices=idx_foo) 
    d_bar = Subset(DS(td, t=bar), indices=idx_bar)
    dl_foo = DL(d_foo, batch_size=3, collate_fn=lambda x:x)
    dl_bar = DL(d_bar, batch_size=3, collate_fn=lambda x:x)

    print('\nFOO-------\n')
    for d in dl_foo:
        for k in d.keys():
            print(f'{k} - {d[k]}')

    print('\nBAR-------\n')
    for d in dl_bar:
        for k in d.keys():
            print(f'{k} - {d[k]}')
