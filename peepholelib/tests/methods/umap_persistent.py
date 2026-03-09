import torch
from tensordict import PersistentTensorDict
from torch.utils.data import DataLoader 
from tensordict import MemoryMappedTensor as MMT

from time import time
import numpy as np

import umap

ds = int(1e2)
ns = int(1e3)
bs = int(1e3)

fname = './banana'

if __name__ == "__main__":
    ''' 
    # create td
    td = PersistentTensorDict(filename=fname, batch_size=[ns], mode='w')
    
    # fill td
    td['a'] = MMT.zeros(shape=(ns,ds))
    dl = DataLoader(td, batch_size=bs, collate_fn=lambda x:x)
    for data in dl:
        data['a'] = torch.rand(data['a'].shape) 
    td.close()
    '''
    ncs = np.linspace(2, ds-1, 4, dtype=np.int32)
    nns = np.linspace(2, ns-1, 4, dtype=np.int32) 
    fitt = torch.zeros(len(ncs), len(nns))
    predt = torch.zeros(len(ncs), len(nns))
    
    d = torch.hstack((
        torch.sin(torch.linspace(0,1,100)).reshape(-1,1),
        torch.cos(torch.linspace(0,1,100)).reshape(-1,1),
        ))
    print(d)
    quit()
    d = torch.rand(ns, ds)
    reducer = umap.UMAP(n_components=nc, n_neighbors=nn)
    t0 = time()
    reducer.fit(d)
    fitt[i][j] = time()-t0

    d = torch.rand(ns, ds)
    t0 = time()
    rd = reducer.transform(d)
    predt[i][j] = time()-t0
            
    print(fitt)
    print(predt)
