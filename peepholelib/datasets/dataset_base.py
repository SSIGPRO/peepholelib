# General python stuff
from pathlib import Path as Path
import abc  

# torch stuff
from tensordict import TensorDict, PersistentTensorDict
from torch.utils.data import random_split, DataLoader

class DatasetBase(metaclass=abc.ABCMeta):

    from .parse_ds import parse_ds

    def __init__(self, **kwargs):
        self.data_path =  Path(kwargs.get('data_path'))
        self.name = kwargs.get('name')

        # computed in load_data()
        self.dss_ = None 
        self._classes = None
    
    @abc.abstractmethod
    def load_data(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get(self):
        raise NotImplementedError()
    
    def get_classes(self):
        if not self._classes:
            raise RuntimeError('Data not loaded. Please run model.load_data() first.')

        return self._classes
    
    def load_only(self, **kwargs):
        '''
        Load already computed dataset.

        Args:
        - loadrs (list[str]): load the specified loaders
        - mode (str): Opens the file with the specified mode. See 'tensordict.PersistentTensorDict.from_h5()' for details. Defaults to 'r'.
        - norm_file (str): load the normalization information. Defaults to None. 
        - verbose (bool): print progress messages.
        '''
        self.check_uncontexted()

        loaders = kwargs.get('loaders')
        mode = kwargs.get('mode', 'r')
        norm_file = kwargs.get('norm_file', None)
        if norm_file != None: norm_file = Path(norm_file)
        verbose = kwargs.get('verbose', False)

        self._dss = {}
        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting data from {ds_key}\n')
            
            _dss_file_paths = self.data_path/('dss.'+ds_key)

            if verbose: print(f'Loading files {_dss_file_paths} from disk. ')
            self._dss[ds_key] = PersistentTensorDict.from_h5(_dss_file_paths, mode=mode)

            _n_samples = len(self._dss[ds_key])
            if verbose: print('loaded n_samples: ', _n_samples)

        return
    
    def __enter__(self):
        self._is_contexted = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        verbose = True 

        if self._dss == None:
            if verbose: print('no dss to close.')
        else:
            for ds_key in self._dss:
                if verbose: print(f'closing {ds_key}')
                self._dss[ds_key].close()

        self._is_contexted = False 
        return

    def check_uncontexted(self):
        if not self._is_contexted:
            raise RuntimeError('Function should be called within context manager')
        return


