from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT

import torch
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader as DL
from torchrl.data import LazyTensorStorage
import torch.multiprocessing as mp

from pathlib import Path

#TODO: check LazyMemmapStorage https://docs.pytorch.org/rl/0.10/reference/generated/torchrl.data.replay_buffers.LazyMemmapStorage.html#torchrl.data.replay_buffers.LazyMemmapStorage
# seems to be faster than LazyTensorStorage, but that is on CPU, on the GPU it might incur unecessary transfers from CPU and GPU

class Storage(LazyTensorStorage):
    def __init__(self, batch_size, device):
        super().__init__(max_size=2*batch_size, device=device)
        self.batch_size = batch_size
        self._occupied = torch.tensor([False, False])
        self._next = False #False for slot 0 and True for slot 1 
        self._ptr = False #False for slot 0 and True for slot 1 
        self._locks = [mp.Lock(), mp.Lock()]
        self._events = [mp.Event(), mp.Event()]

        return

    def __len__(self):
        return int(self._occupied.sum())

    def add(self, data):
        idx = int(self._next)

        print('occ: ', self._occupied)
        print('idx: ', self._next, idx)

        if self._occupied[idx]:
            print("Queue full waiting")
            self._events[idx].wait()
        
        with self._locks[idx]:
            super().set(idx, data)
            self._occupied[idx] = True
            self._next = not self._next
            self._events[idx].set()

        print('occ: ', self._occupied)
        print('next: ', self._next, int(self._next))
        return

    def get(self):
        idx = int(self._ptr)

        print('occ: ', self._occupied)
        print('idx: ', self._ptr, idx)

        if not self._occupied[idx]:
            print("Queue empty waiting")
            self._events[idx].wait()
        
        with self._locks[idx]:
            data = super().get(idx)
            self._occupied[idx] = False
            self._ptr = not self._ptr
            self._events[idx].set()

        print('occ: ', self._occupied)
        print('prt: ', self._ptr, int(self._ptr))

        return data

def p(d):
    for k in d.keys():
        print(k, d[k])

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    dev = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {dev} for computing")

    ds = 3
    ns = 6
    bs = 2
    src = Path('./aaa')
    ks = ['a', 'b']

    #  create src PTD
    if not src.exists():
        td = PTD(filename=src, batch_size=[ns], mode='w')
        for k in ks:
            td[k] = MMT.empty(shape=(ns, ds))
        
        dl = DL(td, batch_size=bs, collate_fn=lambda x:x)
        for d in dl:
            s = len(d)
            for k in d.keys():
                d[k] = torch.rand(s, ds)
        td.close()
        
    td = PTD.from_h5(filename=src, mode = 'r')
    print('DATA:\n')
    for k in td.keys():
        print(k, td[k])
    print('\n')

    st = Storage(
            batch_size = bs,
            device = dev,
            )

    dl = DL(td, batch_size=bs, collate_fn=lambda x:x) 
    it = iter(dl)

    print('\nadding 1: ')
    s = next(it)
    p(s) 
    st.add(s.to(dev, non_blocking=True))

    print('\nadding 2: ')
    s = next(it)
    p(s) 
    st.add(s.to(dev, non_blocking=True))

    print('\ngetting 1: ')
    s = st.get()
    p(s) 

    print('\nadding 3: ')
    s = next(it)
    p(s)
    st.add(s.to(dev, non_blocking=True))

    print('\ngetting 2: ')
    s = st.get()
    p(s) 

    print('\ngetting 3: ')
    s = st.get()
    p(s) 
