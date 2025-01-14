import torchattacks

import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

from pathlib import Path as Path
import abc 

def ftd(data, key):
    return data[key]

class AttackBase(metaclass=abc.ABCMeta):
    
    def __init__(self, **kwargs):
        self.path = Path(kwargs['path'])
        self.name = Path(kwargs['name'])
        self.mode = kwargs['mode'] if 'mode' in kwargs else None
        # computed in load_data()
        self.model = None
        self._loaders = None
        self.res = None
            
    @abc.abstractmethod
    def get_ds_attack(self):
        raise NotImplementedError()
    
