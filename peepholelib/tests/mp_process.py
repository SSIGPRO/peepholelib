from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT

import torch
from torch.utils.data import DataLoader as DL
from cuda_selector import auto_cuda
from torch.multiprocessing import Process, get_context 
from torchrl.data import LazyTensorStorage
import torch.multiprocessing as mp

from pathlib import Path
from time import time

#-----------------------------
# things we learned
#

def p(d):
    for k in d.keys():
        print(k, d[k])

class Storage(LazyTensorStorage):
    def __init__(self, batch_size, device):
        super().__init__(max_size=2*batch_size, device=device, shared_init=True, cleanup_memmap=False)
        self.batch_size = batch_size
        self._next = False #False for slot 0 and True for slot 1 
        self._ptr = False #False for slot 0 and True for slot 1 
        self._locks = [mp.Lock(), mp.Lock()]

        # synching events
        self._free_events = [mp.Event(), mp.Event()]
        [e.set() for e in self._free_events]
        self._fill_events = [mp.Event(), mp.Event()]
        [e.clear() for e in self._fill_events]

        return

    def add(self, data):
        idx = int(self._next)

        self._free_events[idx].wait()
        with self._locks[idx]:
            super().set(idx, data)
            self._next = not self._next

            self._free_events[idx].clear()
            self._fill_events[idx].set()

        return

    def get(self):
        idx = int(self._ptr)

        self._fill_events[idx].wait()
        with self._locks[idx]:
            data = super().get(idx)
            self._ptr = not self._ptr

            self._fill_events[idx].clear()
            self._free_events[idx].set()

        return data

def send(file, bs, device, q, end_event):
    td = PTD.from_h5(filename=file, mode = 'r')
    dl = DL(td, batch_size=bs, collate_fn=lambda x:x)

    for d in dl:
        q.add(d.to(device, non_blocking=True))

    td.close()
    end_event.wait()
    return

def proc(file, bs, q, end_event):
    td = PTD.from_h5(filename=file, mode = 'r+')
    dl = DL(td, batch_size=bs, collate_fn=lambda x:x)

    for d in dl:
        _d = q.get()
        for k in _d.keys():
            _temp = _d[k]
            for i in range(30):
                _temp = _temp@torch.rand(size=_temp.shape).T@torch.rand(size=_temp.shape)
            d[k] = _temp

    td.close()
    end_event.set()
    return

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    dev_cp = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {dev_cp} for computing")

    ns = 2**13#19
    ds = 2**11#13
    bs = 2**9#17
    src = Path('./aaa')
    dst = Path('./bbb')
    dst0 = Path('./ccc')
    ks = ['a']
    
    #  create src PTD
    if not src.exists():
        td_src = PTD(filename=src, batch_size=[ns], mode='w')
        for k in ks:
            td_src[k] = MMT.empty(shape=(ns, ds))
        
        dl = DL(td_src, batch_size=bs, collate_fn=lambda x:x)
        for d in dl:
            s = len(d)
            for k in d.keys():
                d[k] = torch.rand(s, ds)

        print('\n source: ')
        p(td_src)

        td_src.close()

    # dest PTD 
    if not dst.exists():
        td_dst = PTD(filename=dst, batch_size=[ns], mode='w')
        for k in ks:
            td_dst[k] = MMT.empty(shape=(ns, ds))
        td_dst.close()

    # dest PTD reference 
    if not dst0.exists():
        td_dst0 = PTD(filename=dst0, batch_size=[ns], mode='w')
        for k in ks:
            td_dst0[k] = MMT.empty(shape=(ns, ds))
        td_dst0.close()
                                                                
    # Trying with threads 
    t0 = time()
    ctx = get_context("spawn")
    queue = Storage(batch_size=bs, device=dev_cp)
    end_event = ctx.Event()
    p_send = ctx.Process(target=send, args=(src, bs, dev_cp, queue, end_event))
    p_proc = ctx.Process(target=proc, args=(dst, bs, queue, end_event))
    p_send.start()
    p_proc.start()
    p_send.join()
    p_proc.join()
    print('new time: ', time()-t0) 

    # compute ref
    td_src = PTD.from_h5(filename=src, mode = 'r')
    td_dst0 = PTD.from_h5(filename=dst0, mode = 'r+')
    dl_src = DL(td_src, batch_size=bs, collate_fn=lambda x:x)
    dl_dst0 = DL(td_dst0, batch_size=bs, collate_fn=lambda x:x)
    t0 = time()
    for d_s, d_t0 in zip(dl_src, dl_dst0):
        for k in ks:
            _temp = d_s[k]
            for i in range(30):
                _temp = _temp@torch.rand(size=_temp.shape).T@torch.rand(size=_temp.shape)
            d_t0[k] = _temp
    print('old time: ', time()-t0) 
    td_src.close()
    td_dst0.close()
                                                                
    # check correctness 
    td_dst = PTD.from_h5(filename=dst, mode = 'r+')
    td_dst0 = PTD.from_h5(filename=dst0, mode = 'r+')
    dl_dst = DL(td_dst, batch_size=bs, collate_fn=lambda x:x)
    dl_dst0 = DL(td_dst0, batch_size=bs, collate_fn=lambda x:x)
    for d_t, d_t0 in zip(dl_dst, dl_dst0):
        corr = True
        for k in ks:
            corr = corr and (d_t[k] == d_t0[k]).all()
        print(f'batch correct: {corr}')

    print('\n ref: ')
    p(td_dst0)

    print('\n new: ')
    p(td_dst)

    # close
    td_dst.close()
    td_dst0.close()
