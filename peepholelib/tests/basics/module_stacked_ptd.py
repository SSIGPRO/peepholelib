from tensordict import PersistentTensorDict as PTD
from tensordict import TensorDict as TD
from tensordict.functional import merge_tensordicts
from tensordict import MemoryMappedTensor as MMT
from tensordict import merge_tensordicts as mtd 
from tensordict import LazyStackedTensorDict as LSTD 

import torch
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader as DL
from torch.utils.data import Dataset 

from pathlib import Path

def p(d):
    for k in d.keys():
        print(k, d[k])

class _ModuleWiseStack:
    def __init__(self, **kwargs):
        _tds = kwargs['tds']
        self.tds = {}

        # concatenate the keys, those should be mutually exclusive
        # and have the same length
        self.k = []
        lens = []
        for td in _tds:
            keys = list(td.keys())
            self.k += keys
            lens.append(len(td[keys[0]]))
        
        if len(set(self.k)) != len(self.k):
            raise RuntimeError(f'PTDs should have mutually exclusive keys. Got: {self.k}')

        if len(set(lens)) > 1:
            raise RuntimeError(f'PTDs should have the same lenght. Got: {lens}')

        for td in _tds:
            for k in list(td.keys()):
                self.tds[k] = td

        self.len = lens[0]
        return

    def __getitem__(self, idx):
        if type(idx) == str:
            return self.tds[idx][idx]
        else:
            r = {}
            for k in self.tds.keys():
                r[k] = self.tds[k][k][idx] 
            return r 
    
    def __getitems__(self, idx):
        r = TD({}, batch_size=len(idx))
        for k in self.tds.keys():
            r[k] = self.tds[k][k][idx] 

        return r 

    def __len__(self):
        return self.len

    def keys(self):
        return self.k
    
    def mean(self, dim=0):
        r = TD({})
        for k in self.tds.keys():
            r[k] = self.tds[k][k].mean(dim=dim) 
        return r

    def std(self, dim=0):
        r = TD({})
        for k in self.tds.keys():
            r[k] = self.tds[k][k].std(dim=dim) 
        return r                                    

    def close(self):
        for k in self.tds.keys():
            self.tds[k].close()
        return

class DS(Dataset):
    def __init__(self, d, t):
        self.len = len(d)
        self.d = d
        self.t = t

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {'data': self.d[idx], 'label': self.t[idx]}

    def __getitems__(self, idx):
        return {'data': self.d[idx], 'label': self.t[idx]}

def p(td, bs):
    dl = DL(td, batch_size=bs, collate_fn=lambda x:x)
    for d in dl:
        for k in d.keys():
            print(f'{k} - {d[k]}')
    return

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    dev = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f"Using {dev} for computing")

    ns = 2**12
    ds = 2**12
    bs = 2**1
    srca = Path('./aaa')
    srcb = Path('./bbb')

    #  create src PTD
    dsa = DS(d=torch.rand(ns, ds), t=torch.randint(0, 10, size=(ns,1)))

    #  create src PTD
    if not srca.exists():
        tda = PTD(filename=srca, batch_size=[ns], mode='w')
        tda['a'] = MMT.empty(shape=(ns, ds))
        
        dl = DL(tda, batch_size=bs, collate_fn=lambda x:x)
        for d in dl:
            s = len(d)
            d['a'] = torch.rand(s, ds)
    else:
        tda = PTD.from_h5(filename=srca, mode = 'r')

    if not srcb.exists():
        tdb = PTD(filename=srcb, batch_size=[ns], mode='w')
        tdb['b'] = MMT.empty(shape=(ns, ds))
        
        dl = DL(tdb, batch_size=bs, collate_fn=lambda x:x)
        for d in dl:
            s = len(d)
            d['b'] = torch.rand(s, ds)
    else:
        tdb = PTD.from_h5(filename=srcb, mode = 'r')
    
    input('wait')
    tds = _ModuleWiseStack(tds=[tda, tdb])
    input('wait')
    p(tda, bs)
    p(tdb, bs)
    p(tds, bs)
    m = tds.mean(dim=0)
    s = tds.std(dim=0)
    tda.close()
    tdb.close()
    tds.close()

