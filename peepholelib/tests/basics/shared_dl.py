from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
import torch
import sys
from torch.utils.data import DataLoader
import time
import multiprocessing
import threading

class foo():
    def __init__(self, dl, n):
        self.dl = dl
        self.n = n

    def do(self):
        for d in self.dl:
            print(f'p{self.n} - td[a0]: ', d['a0'])
            time.sleep(1)
        return

def p(td):
    for k in td.keys():
        print(f'td[{k}] = ', td[k])

if __name__ == '__main__':
    n = 5 
    bs = 2
    ds = 3
    device = 'cpu' 
    n_dicts = 1 
    
    # seems like batch size is really important
    td = TensorDict(batch_size=n, device=device)
    for j in range(n_dicts):
        td['a%d'%j] = MMT.empty(shape=(n,ds))
    
    td = td.memmap_like('./banana')
    # makes one dict for each input
    for j in range(n_dicts):
        td['a%d'%j][:] = torch.rand((n, ds))
    
    print('saving', sys.getsizeof(td))
    p(td) 

    dl = DataLoader(td, batch_size=bs, collate_fn=lambda x:x)
    print('\n ---- \n DataLoading\n')
    for d in dl:
        for j in range(n_dicts):
            print('td[a%d]: '%j, d['a%d'%j])
    print('\n ----')

    f0 = foo(dl, '0')
    f1 = foo(dl, '1')
    
    p0 = multiprocessing.Process(name='0', target=f0.do)
    p1 = multiprocessing.Process(name='1', target=f1.do)
    
    print('\n ---- process \n')
    p0.start()
    p1.start()
    
    time.sleep(5)

    t0 = threading.Thread(target=f0.do, args={})
    t1 = threading.Thread(target=f1.do, args={})
    print('\n ---- threads \n')
    t0.start()
    t1.start()
    t0.join()
    t1.join()
