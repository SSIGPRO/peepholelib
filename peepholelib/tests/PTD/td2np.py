from tensordict import MemoryMappedTensor as MMT
from pathlib import Path
import torch
import numpy as np

f = Path.cwd()/'bananat' 

a = MMT.from_tensor(torch.rand(10), filename=f)
b = np.memmap(f, mode='r', shape=(10,), dtype=np.float32)
print(a)
print(b)

c = np.memmap(f.as_posix()+'t2', mode='w+', shape=(10,), dtype=np.float32)
c[:] = a[:]
print(c)
