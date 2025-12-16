# python stuff
from pathlib import Path
from tqdm import tqdm
from math import ceil

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

from peepholelib.peepholes import drill_base as driller

class Peepholes:
    def __init__(self, **kwargs):
        self.path = Path(kwargs['path'])
        self.name = kwargs['name']
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        # Set in get_peepholes() 
        self.target_modules = None # list of peep modules
        self._drillers = None 

        # computed in get_peepholes
        self._phs = {} 
        
        # computed in get_dataloaders()
        self._loaders = None

        # Set on __enter__() and __exit__()
        # read before each function
        self._is_contexted = False

        return

    def get_peepholes(self, **kwargs):
        '''
        Compute peepholes given `corevectors` and `drillers`.
        
        Args:
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): Parsed datasets respective the `coreVectors`.
        - corevectors (peepholelib.coreVectors.coreVectors.coreVectors): corevectors respective the `datasets`.
        - loaders (list[str]): list of loaders, usually `['train', 'val', 'test']`. If `None` uses all loaders in `corevectors._corevds.keys()`. Defaults to dss `None`.
        - target_modules (list[str]): list of modules to consider as in `model.state_dict`.
        - drillers (dict(str: peepholelib.peepholes.drill_base.DrillBase)):Dictionary where keys are the modules as in `model.state_dict` and values are classes extending `DrillBase`.
        - batchsize (int): batchsize to process `corevectors` into `peepholes`. Defaults to 64.
        - n_threads (int): Number of threads to pass as `num_workers` to `torch.utils.data.DataLoader`. Defaults to 1.
        - verbose (bool): print progress messages
        '''
        self.check_uncontexted()
        
        datasets = kwargs['datasets']
        corevectors = kwargs['corevectors']
        loaders  = kwargs.get('loaders', None)
        self.target_modules = kwargs['target_modules'] # list of peep modules
        self._drillers = kwargs['drillers']

        bs = kwargs.get('batch_size', 64)
        n_threads = kwargs.get('n_threads', 1)
        verbose = kwargs.get('verbose', False)

        if loaders == None: loaders = list(corevectors._corevds.keys())

        for ds_key in loaders:
            cvds = corevectors._corevds[ds_key]
            dssds = datasets._dss[ds_key]

            if verbose: print(f'\n ---- Getting peepholes for {ds_key}\n')
            file_path = self.path/(self.name+'.'+ds_key)
            
            # create/load PersistentTensorDict file
            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
                n_samples = len(self._phs[ds_key])
                if verbose: print('loaded n_samples: ', n_samples)
            else:
                n_samples = len(cvds)
                if verbose: print('loader n_samples: ', n_samples) 
                self.path.mkdir(parents=True, exist_ok=True)
                self._phs[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
            
            modules_to_compute = []
            for module in self.target_modules:
                if not module in self._phs[ds_key]:
                    #------------------------
                    # Pre-allocate peepholes
                    #------------------------
                    if verbose: print('allocating peepholes for module: ', module)
                    self._phs[ds_key][module] = MMT.empty(shape=(n_samples, self._drillers[module].nl_model))
                    modules_to_compute.append(module)
                else:
                    if verbose: print(f'Peepholes for {module} already present. Skipping.')

            # Close PTD create with mode 'w' and re-open it with mode 'r+'
            # This is done so we can use multiple workers for reading and writting
            self._phs[ds_key].close()
            self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            #------------------------ 
            # computing peepholes
            #------------------------
            # create dataloaders
            dl_phs = DataLoader(self._phs[ds_key], batch_size=bs, collate_fn=lambda x:x, num_workers = n_threads)
            dl_cvs = DataLoader(cvds, batch_size=bs, collate_fn=lambda x: x, num_workers = n_threads)
            dl_dss = DataLoader(dssds, batch_size=bs, collate_fn=lambda x: x, num_workers = n_threads)

            if len(modules_to_compute) == 0:
                if verbose: print(f'No modules to compute for {ds_key}. Skipping.')
                continue

            if verbose: print(f'\n ---- computing peepholes for modules {modules_to_compute}\n')
            for _cvs, _dss, phs in tqdm(zip(dl_cvs, dl_dss, dl_phs), disable=not verbose, total=ceil(n_samples/bs)):
                for module in modules_to_compute:
                    phs[module] = self._drillers[module](cvs=_cvs[module], dss=_dss)

        return 

    def load_only(self, **kwargs):
        '''
        Load the peepholes 
        '''
        self.check_uncontexted()

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        loaders = kwargs['loaders']
        mode = kwargs['mode'] if 'mode' in kwargs else 'r'

        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting peepholes for {ds_key}\n')
            file_path = self.path/(self.name+'.'+ds_key)
           
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode=mode)

        return

    def get_conceptograms(self, **kwargs):
        '''
        Get conceptograms from peepholes. A conceptogram is the concatenation of peepholes for multiple modules.
        
        Args:
        - target_modules (list[str]): list of target module keys
        - loaders (list[str]): list of loaders (usually 'train', 'test', 'val' within self._phs 
        - verbose (bool): print progress information
        '''
        self.check_uncontexted()
        
        target_modules = kwargs.get('target_modules', None)
        verbose = kwargs.get('verbose', False)

        if self._phs == None:
            raise RuntimeError('Peepholes not present. Please run get_peepholes() first.')

        loaders = kwargs.get('loaders', list(self._phs))

        _conceptograms = {}
        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting conceptograms for {ds_key}\n')
            file_path = self.path / (self.name + '.' + ds_key)

            #-----------------------------------------
            # Check if peepholes exist before computing scores
            #-----------------------------------------
            n_samples = len(self._phs[ds_key])

            if target_modules == None:
                target_modules = self._phs[ds_key].keys()

            for module in target_modules:
                if module not in self._phs[ds_key]:
                    raise ValueError(f"Peepholes for module {module} do not exist. Please run get_peepholes() first.")

            _conceptograms[ds_key] = torch.stack([self._phs[ds_key][layer] for layer in target_modules], dim=1)

        return _conceptograms

    def __enter__(self):
        self._is_contexted = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        verbose = True 

        if self._phs == None:
            if verbose: print('no peepholes to close. doing nothing.')
            return

        for ds_key in self._phs:
            if verbose: print(f'closing {ds_key}')
            self._phs[ds_key].close()
            
        self._is_contexted = False 
        return

    def check_uncontexted(self):
        if not self._is_contexted:
            raise RuntimeError('Function should be called within context manager')
        return
