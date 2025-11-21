# General python stuff
from pathlib import Path as Path
import abc  

class DatasetWrap(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        '''
        Creates instance of dataset base. For base datasets, `Transform` is mandatory.

        Args:
        - path (str): Path for dataset. 
        - seed (int): Random seed for reproducibility.
        - transform (torchvision.transforms.Compose): Custom transform to apply to the original dataset

        '''
        self.path = Path(kwargs.get('path'))
        self.seed = kwargs.get('seed', 42)
        self.transform = kwargs['transform']
        
        return

    @abc.abstractmethod
    def __load_data__(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get(self, ds_key, idx):
        raise NotImplementedError()
