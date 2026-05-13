# python stuff
from pathlib import Path
from tqdm import tqdm
from math import ceil

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

# our stuff
from peepholelib.peepholes import drill_base as driller
from peepholelib.utils.ptd_wraps import _ModuleWiseStack

class Peepholes:
    def __init__(self, **kwargs):
        '''
        Args:
        - path (str|pathlib.Path): Path to save corevectors.
        '''
        self.path = Path(kwargs['path'])

        # Set in get_peepholes() or load_only() 
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
        - drillers (dict(str: peepholelib.peepholes.drill_base.DrillBase)):Dictionary where keys are the modules as in `model.state_dict` and values are classes extending `DrillBase`.
        - names dict(str:str): Dictionary with key being the module name, and value being a name to append to the PTD file with the peepholes. Peepholes will be saved in a file with name `<loader>/<key>.<name>. If `None` it is ignored. Defaults to `None`.
        - batchsize (int): batchsize to process `corevectors` into `peepholes`. Defaults to 64.
        - n_threads (int): Number of threads to pass as `num_workers` to `torch.utils.data.DataLoader`. Defaults to 1.
        - verbose (bool): print progress messages
        '''
        self.check_uncontexted()

        dss = kwargs['datasets']
        cvs = kwargs['corevectors']
        loaders  = kwargs.get('loaders', None)
        self._drillers = kwargs['drillers']
        names = kwargs.get('names', None)
        bs = kwargs.get('batch_size', 64)
        n_threads = kwargs.get('n_threads', 1)

        verbose = kwargs.get('verbose', False)
        target_modules = kwargs['target_modules'] # list of peep modules

        if loaders == None: loaders = list(cvs._corevds.keys())

        for ds_key in loaders:
            #------------------------
            # Pre-allocate peepholes
            #------------------------
            if verbose: print(f'\n ---- Getting peepholes for {ds_key}\n')

            _tds = {}
            _mtc = [] #list of modules to compute
            for mk in cvs._corevds[ds_key].keys(): 
                if names == None:
                    file_path = self.path/ds_key/mk
                else:
                    file_path = self.path/ds_key/(mk+'.'+names[mk])

                file_path.parent.mkdir(parents=True, exist_ok=True)

                # create/load PersistentTensorDict file
                if file_path.exists():
                    if verbose: print(f'File {file_path} exists. Loading from disk.')
                    _td = PersistentTensorDict.from_h5(file_path, mode='r')
                    n_samples = len(_td)
                else:
                    n_samples = len(cvs._corevds[ds_key])
                    if verbose: print(f'Allocating peepholes for module {mk} with {n_samples} samples.')
                    _td = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
                    # dry run to get size and dtype
                    _cv = cvs._corevds[ds_key][mk][0:1]
                    _d = dss._dss[ds_key][0:1] 
                    _ph = self._drillers[mk](cvs=_cv, dss=_d)

                    # allocate peepholes 
                    _td[mk] = MMT.empty(shape=(n_samples,)+_ph.shape[1:], dtype=_ph.dtype)
                    _mtc.append(mk)

                    # close and open it again to use with the dataloaders
                    _td.close()
                    _td = PersistentTensorDict.from_h5(file_path, mode='r+')

                _tds[mk] = _td

            #------------------------ 
            # compute peepholes
            #------------------------
            if len(_mtc) == 0:
                if verbose: print(f'No modules to compute for {ds_key}. Skipping.')
                self._phs[ds_key] = _ModuleWiseStack(tds=_tds)
                continue

            if verbose: print(f'\n ---- computing peepholes for modules {_mtc}\n')

            # create dataloaders
            dss_dl = DataLoader(dss._dss[ds_key], batch_size=bs, collate_fn=lambda x: x, num_workers = n_threads)
            cvs_dl = DataLoader(cvs._corevds[ds_key], batch_size=bs, collate_fn=lambda x: x, num_workers = n_threads)

            phs_dls = [
                    DataLoader(_tds[mk], batch_size=bs, collate_fn=lambda x:x, num_workers = n_threads) for mk in _mtc
                    ]

            for data in tqdm(zip(dss_dl, cvs_dl, *phs_dls), disable=not verbose, total=ceil(n_samples/bs)):
                # the first data in the tuple is the dataset
                # the second is the corevectors 
                # to next ones are from phs_dls, ordered according to _mtc
                _dss = data[0]
                _cvs = data[1]
                phs = {mk: data[i+2] for i, mk in enumerate(_mtc)}
                for mk in _mtc:
                    phs[mk][mk] = self._drillers[mk](cvs=_cvs[mk], dss=_dss)
            
            # save as stacked
            self._phs[ds_key] = _ModuleWiseStack(tds=_tds)

        return 

    def load_only(self, **kwargs):
        '''
        Load the peepholes 
        '''
        self.check_uncontexted()

        loaders = kwargs['loaders']
        names = kwargs['names']
        mode = kwargs.get('mode', 'r')
        verbose = kwargs.get('verbose', False)

        self.__close()

        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting peepholes for {ds_key}\n')

            _tds = {}
            for mk in names.keys():
                if names[mk] == None:
                    file_path = self.path/ds_key/mk
                else:
                    file_path = self.path/ds_key/(mk+'.'+names[mk])

                if verbose: print(f'Loading file {file_path}. ')
                _td = PersistentTensorDict.from_h5(file_path, mode=mode)
                _tds[mk] = _td

            self._phs[ds_key] = _ModuleWiseStack(tds=_tds)
            if verbose: print('loaded n_samples: ', len(self._phs[ds_key]))

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

        if self._phs == {}:
            raise RuntimeError('Peepholes not present. Please run get_peepholes() first.')

        loaders = kwargs.get('loaders', list(self._phs))

        _conceptograms = {}
        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting conceptograms for {ds_key}\n')

            #-----------------------------------------
            # Check if peepholes exist before computing scores
            #-----------------------------------------
            n_samples = len(self._phs[ds_key])

            if target_modules == None:
                target_modules = self._phs[ds_key].keys()

            for module in target_modules:
                if module not in self._phs[ds_key].keys():
                    raise ValueError(f"Peepholes for module {module} do not exist. Please run get_peepholes() first.")

            _conceptograms[ds_key] = torch.stack([self._phs[ds_key][layer] for layer in target_modules], dim=1)

        return _conceptograms

    def __close(self):
        if self._phs != {}:
            for ds_key in self._phs:
                self._phs[ds_key].close()

        # reset these
        self._phs = {}
        return

    def __enter__(self):
        self._is_contexted = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__close()    
        self._is_contexted = False 
        return

    def check_uncontexted(self):
        if not self._is_contexted:
            raise RuntimeError('Function should be called within context manager')
        return
