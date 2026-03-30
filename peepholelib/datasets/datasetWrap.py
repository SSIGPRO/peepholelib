# General python stuff
from pathlib import Path as Path
import abc  

class DatasetWrap(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        '''
        Creates instance of dataset base. For base datasets.

        Args:
        - path (str): Path for dataset. 
        - seed (int): Random seed for reproducibility.

        '''
        self.path = Path(kwargs['path'])
        self.seed = kwargs.get('seed', 42)

        # useful for checking in `datasets.parsedDatase.ParsedDataset` 
        self.has_transforms = ('std_transform' in kwargs) or ('aug_transform' in kwargs)
        
        return

    @abc.abstractmethod
    def __load_data__(self):
        raise NotImplementedError()

