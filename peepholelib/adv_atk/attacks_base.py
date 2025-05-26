import torchattacks

import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

from peepholelib.datasets.dataset_base import DatasetBase

from pathlib import Path as Path
import abc 

def ftd(data, key_list):
    r = {}
    for k in key_list:
        r[k] = data[k]
    return r 

class AttackBase(DatasetBase):
    
    def __init__(self, **kwargs):

        DatasetBase.__init__(self, **kwargs)
        self.path = Path(kwargs['path'])
        self.name = Path(kwargs['name'])
        self.mode = kwargs['mode'] if 'mode' in kwargs else None
        
        self.model = None
        self._loaders = None
        self.res = None

    
    def load_data(self, **kwargs):
        if not self.atk_path.exists(): raise RuntimeError(f'Attack path {self.atk_path} does not exist. Please run get_ds_attack() first.')
        self._dss = {}
        if self.verbose: print(f'File {self.atk_path} exists.')
        for ds_key in self._loaders:
            self._dss[ds_key] = TensorDict.load_memmap(self.atk_path/ds_key)


    @abc.abstractmethod
    def get_ds_attack(self):
        raise NotImplementedError()
    
