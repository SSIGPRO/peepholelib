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
        self.path = Path(kwargs['data_path'])
        self.name = Path(kwargs['name'])
        self.mode = kwargs['mode'] if 'mode' in kwargs else None
        
        self.model = None
        self._loaders = None
        self.res = None

    
    def load_data(self, **kwargs):
        if not self.data_path.exists(): raise RuntimeError(f'Attack path {self.data_path} does not exist. Please run get_ds_attack() first.')
        print(self.data_path)
        self._dss = {}
        if self.verbose: print(f'File {self.data_path} exists.')
        for ds_key in self._loaders:
            self._dss[ds_key] = TensorDict.load_memmap(self.data_path/ds_key)

    @abc.abstractmethod
    def get_ds_attack(self):
        raise NotImplementedError()
    
    def get(self, ds_key, idx):
        '''
        Get item from the dataset.
        
        Args:
        - idx (int): Index of the item to get.
        - ds_key (str): Key of the dataset to get the item from ('train', 'val', 'test').
        
        Returns:
        - a tuple of (image, label)
        '''
        if not self._dss:
            raise RuntimeError('Data not loaded. Please run load_data() first.')
        data = {'image': self._dss[ds_key]['image'][idx].unsqueeze(0),
                'label': self._dss[ds_key]['label'][idx].unsqueeze(0),
                'attack_success': self._dss[ds_key]['attack_success'][idx].unsqueeze(0)}
        return data
    
