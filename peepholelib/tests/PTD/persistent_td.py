from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
import torch
import sys
from torch.utils.data import DataLoader
from time import time
import tempfile

p = 6 
n = 2**p 
bs = 2**(p-1)-10 
ds = 2**p
nt = 5 
device = 'cuda:1'#'cpu' 
fname = './banana'

def test1(td):
    # num_workers break the stuff
    dl = DataLoader(td, batch_size=bs, collate_fn=lambda x:x)#, num_workers = nt)
    for bn, data in enumerate(dl):
        print('device: ', data.device)
        n_in = data.shape[0]
        print('\n----\nn_in: ', n_in, bn, bs)
        input('wait')
        print('data shape: ', data.shape)
        t0 = time()

        # This one does not work anymore
        #d['a'][bn*bs:bn*bs+n_in] = torch.rand(n_in, ds)

        # use batch directly
        data['a'] = torch.rand(n_in, ds) 

        print('time: ', time()-t0)
    print('\n printing some values filling:\n', td['a'][-3:-1,-3:-1])

# using mem_map has the problem of increasing memory
# seems it makes an in mem copy
def meta_test1():
    td = TensorDict(batch_size=n, device=device)
    print('allocating: ', n, ds)
    td['a'] = MMT.zeros(shape=(n,ds))
    td = td.memmap_like(fname, num_threads=nt)
    input('mapped') 
    test1(td) 

def ab(xx):
    dl = DataLoader(xx, batch_size=bs, collate_fn=lambda x:x)
    for bn, data in enumerate(dl):
        print('device: ', data.device)
        n_in = data.shape[0]
        print('\n----\nn_in: ', n_in, bn, bs)
        input('wait')
        print('data shape: ', data.shape)
        t0 = time()
        data['b']['b0'] = torch.hstack((data['a']+1, torch.rand(n_in,2).to(device)))
        data['b']['b1'] = torch.hstack((data['a']+2, torch.rand(n_in,3).to(device)))

        print('time: ', time()-t0)

def meta_test2():
    td = PersistentTensorDict(filename=fname, batch_size=[n], device=device, mode='w')
    
    print('allocating: ', n, ds)
    td['a'] = MMT.zeros(shape=(n,ds))
    # save after writting
    test1(td)
    print('\n printing some values:\n')
    print(td['a'][-3:-1,-5:-1])
    # do not forget to close
    input('reopening')
    td.close()

    del td
    input('reloading')
    td2 = PersistentTensorDict.from_h5(fname, mode='r+').to(device)
    print('\n printing some values on td2:\n')
    print(td2['a'][-3:-1,-5:-1])
    
    # loading messes up the batch size, needs to correct here
    print('after loading bs: ', td2.batch_size) 
    td2.batch_size = torch.Size((td2.batch_size[0],)) 
    
    print('allocating')
    td2['b'] = TensorDict(batch_size=n)
    td2['b']['b1'] = MMT.zeros(shape=(n,ds+3))
    td2['b']['b0'] = MMT.zeros(shape=(n,ds+2))
    ab(td2)
    print('\n printing some values on td2 - b:\n')
    print(td2['b']['b0'][-3:-1,-5:-1])
    print(td2['b']['b1'][-3:-1,-5:-1])
    td2.close() 
    
    input('re-reloading')
    del td2
    td3 = TensorDict.from_h5(fname, mode='r+').to(device)
    print('\n printing some values on td3 - b:\n')
    print(td3['a'][-3:-1,-5:-1])
    print(td3['b']['b0'][-3:-1,-5:-1])
    print(td3['b']['b1'][-3:-1,-5:-1])
    print('device: ', td3.device)
    td3.close() 

if __name__ == '__main__':
    meta_test2()
