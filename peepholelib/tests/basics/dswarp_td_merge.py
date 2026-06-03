from tensordict import PersistentTensorDict as PTD
from tensordict import TensorDict as TD
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

class UDSWTD:
    def __init__(self, **kwargs):
        self.ori = kwargs['ori']
        self.inf = kwargs['inf']

        s_ori = self.ori[0]
        s_inf = self.inf[0]

        if len(set(s_ori.keys()) & set(s_inf.keys())) > 0:
            raise RuntimeError('Original dataset and parsed inference values should have exclusive sets of keys.')

        if len(self.ori) != len(self.inf):
            raise RuntimeError('Original dataset has {len(self.ori)} samples, parsed inference values have {len(self.inf)} samples. They should have the same number of samples.')

        self.k = list(s_ori.keys()) + list(s_inf.keys())
        self.len = len(self.ori)
        return

    def __getitem__(self, idx):
        if type(idx) == str:
            return self.key_map[idx][idx]
        else:
            r = TD({}, batch_size=len(idx))
            for k, v in self.ori[idx].items():
                r[k] = v 
            for k, v in self.inf[idx].items():
                r[k] = v 
            return r 
    
    def __getitems__(self, idx):
        r = TD({}, batch_size=len(idx))
        for k in self.ori[idx].items():
            r[k] = v 
        for k in self.inf[idx].items():
            r[k] = v 

        return r 

    def __len__(self):
        return self.len

    def keys(self):
        return self.k

    def close(self):
        if isinstance(self.ori, PTD):
            self.ori.close()
        self.inf.close()

class DS(Dataset):
    def __init__(self, d, t):
        self.len = len(d)
        self.d = d
        self.t = t

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {'data': self.d[idx], 'label': self.t[idx]}

    def __getitems_(self, idx):
        return {'data': self.d[idx], 'label': self.t[idx]}

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    dev = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f"Using {dev} for computing")

    ns = 2**15
    ds = 2**15
    bs = 2**13
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
    input('Stackin PTD and PTD')
    tds = UDSWTD(ori=tda, inf=tdb) 
    input('wait')
    dl = DL(tds, batch_size=bs, collate_fn=lambda x:x)
    for d in dl:
        for k in d.keys():
            print(f'{k} - {d[k]}')
    tds.close()

    input('wait')
    input('Stackin DWS and PTD')
    tdb = PTD.from_h5(filename=srcb, mode = 'r')
    tds = UDSWTD(ori=dsa, inf=tdb) 
    input('wait')
    dl = DL(tds, batch_size=bs, collate_fn=lambda x:x)
    for d in dl:
        for k in d.keys():
            print(f'{k} - {d[k]}')
    tds.close()

