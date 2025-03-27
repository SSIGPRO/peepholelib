# General python stuff
from pathlib import Path as Path
import abc  

# torch stuff
from torch.utils.data import random_split, DataLoader

class DatasetBase(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.data_path = Path(kwargs['data_path']) if 'data_path' in kwargs else Path.cwd().parent/'data/datasets'

        # computed in load_data()
        self._dss = None
        self._classes = None
    
    @abc.abstractmethod
    def load_data(self):
        raise NotImplementedError()
    
    def get_classes(self):
        if not self._classes:
            raise RuntimeError('Data not loaded. Please run model.load_data() first.')

        return self._classes

