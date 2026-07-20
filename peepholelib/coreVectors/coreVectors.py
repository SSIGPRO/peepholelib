# generic python stuff
from pathlib import Path
from tqdm import tqdm
from math import ceil
from time import sleep

# torch stuff
import torch
from torch.utils.data import DataLoader
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT

# our stuff
from peepholelib.models.model_wrap import get_in_activations
from peepholelib.utils.ptd_wraps import _ModuleWiseStack

class CoreVectors():
    def __init__(self, **kwargs):
        '''
        Args:
        - path (str|pathlib.Path): Path to save corevectors.
        '''
        # create folder
        self.path = Path(kwargs['path'])
        self.path.mkdir(parents=True, exist_ok=True)

        self._model = kwargs['model'] if 'model' in kwargs else None  

        # computed in get_coreVectors() or load_only()
        self._corevds = {} 
        self._corevds_files = {} 

        # set in normalize_corevectors() or load_only() 
        self._normalizations = {} 
        
        # Set on __enter__() and __exit__()
        # read before each function
        self._is_contexted = False

        # computed in get_dataloaders()
        self._loaders = {}
        return

    def get_coreVectors(self, **kwargs):
        '''
        Compute and save corevectos. Corevectors are saved directly on disk using a 'tensordict.PersistentTensorDict' at 'self.path/self.name.<loader>', with 'loader' being the loader keys (see peepholelib.datasets).
        Pre-allocation is done with shapes obtained via a dry-run. Checks are performed for existing loaders and existing modules, which are skipped.
        If activations are present in 'datasets._dss', use the saved values, but saving activations is memory heavy. Otherwise, pass the input images through the model in batches and get the activations directly from the model (see 'peepholelib.model_wrap').
    
        - datasets (dict(str: peepholelib.datasets.parsedDataset.ParsedDataset)): Parsed datasets.
        - loaders (list[str]): List of loaders in `datasets.keys()` to compute corevectors. If `None` uses `datasets._dss.keys()`. Defaults to `None`.
        - input_key (str): use this key from `datasets` as inputs to the `reducers`. Defaults to `'image'`. 
        - reducers (dict(str: Callable)): A dictionary with keys being the module names as per the model's state_dict, and values being a callable foo(acts) which takes as input the model's batched activations and returns a dimentionality reduced version of its outputs. 
        - activations_parser (callable): A function for parsing activations. Defaults to 'get_in_activations()' (see peepholelib.models.model_wrap.py for details on how we get the activations).
        - names dict(str:str): Dictionary with key being the module name, and value being a name to append to the PTD file with the corevectors. Corevectors will be saved in a file with name `<loader>/<key>.<name>. If `None` it is ignored. Defaults to `None`.
        - batch_size (int): Creates dataloader to do computation in batch size. Defaults to 64.
        - retry_load_time (int): Time (in seconds) to wait before retrying loading an already existing PTD with corevectors. If `None` no further attempts are done. Defaults to `None`.
        - n_threads (int): 'num_workers' passed to 'torch.utils.data.DataLoader'. Defaults to 1.
        - verbose (bool): print progress messages.
        '''
        self.check_uncontexted()

        datasets = kwargs.get('datasets')
        loaders = kwargs.get('loaders', None)
        input_key = kwargs.get('input_key','image')
        reducers = kwargs.get('reducers')
        activations_parser = kwargs.get('activations_parser', get_in_activations)
        names = kwargs.get('names', None)
        bs = kwargs.get('batch_size', 64)
        rlt = kwargs.get('retry_load_time', None)
        n_threads = kwargs.get('n_threads', 1)
        save_input = kwargs.get('save_input', True)
        save_output = kwargs.get('save_output', False)
        verbose = kwargs.get('verbose', False)
    
        model = self._model 
        device = self._model.device 
    
        if reducers.keys() != model._target_modules.keys(): 
            raise RuntimeError(f'Keys inconsistency between reducers and target_modules \n reducers keys: {reducers.keys()} \n target_modules: {model._target_modules.keys()}')
    
        if names != None and names.keys() != model._target_modules.keys(): 
            raise RuntimeError(f'Keys inconsistency between names and target_modules \n names keys: {names.keys()} \n target_modules: {model._target_modules.keys()}')

        # set the model to get activations
        model.set_activations(save_input=save_input, save_output=save_output)
    
        for ds_key in datasets._dss:
            #------------------------------------------------
            # pre-allocate corevectors
            #------------------------------------------------
            if verbose: print(f'\n ---- Getting core vectors for {ds_key}\n')

            # sample for dry run
            with torch.no_grad():
                model(datasets._dss[ds_key][0:1][input_key].to(device))
                _act0 = activations_parser(model._acts)

            _tds = {}
            self._corevds_files[ds_key] = {}
            _mtc = [] #list of modules to compute
            for mk in model._target_modules.keys():
                if names == None:
                    file_path = self.path/ds_key/mk
                else:
                    file_path = self.path/ds_key/(mk+'.'+names[mk])
                self._corevds_files[ds_key][mk] = file_path

                file_path.parent.mkdir(parents=True, exist_ok=True)

                if file_path.exists():
                    if verbose: print(f'File {file_path} exists. Loading from disk.')

                    if rlt == None:
                        _td = PersistentTensorDict.from_h5(file_path, mode='r')
                    else:
                        while True:
                            try:
                                _td = PersistentTensorDict.from_h5(file_path, mode='r')
                                break
                            except BlockingIOError:
                                if verbose: print(f'Seems like the file {file_path} is busy. Will wait {rlt} seconds and try again.')
                                sleep(rlt)

                    n_samples = len(_td)
                else:
                    n_samples = len(datasets._dss[ds_key])
                    if verbose: print(f'Allocating core vectors for module {mk} with {n_samples} samples')
                    _td = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
                    # allocate corevectors 
                    _cv = reducers[mk](act_data=_act0[mk]) # dry run
                    _td[mk] = MMT.empty(shape=((n_samples,)+_cv.shape[1:]), dtype=_cv.dtype)
                    _mtc.append(mk)

                # close and open it again to use with the dataloaders
                _td.close()
                _td = PersistentTensorDict.from_h5(file_path, mode='r+')

                _tds[mk] = _td

            # ---------------------------------------
            # compute corevectors 
            # ---------------------------------------
            if len(_mtc) == 0:
                print(f'No new core vectors for {ds_key}, skipping')
                self._corevds[ds_key] = _ModuleWiseStack(tds=_tds)
                continue
    
            if verbose: print(f'\n ---- Getting corevectors for {ds_key}\n')
    
            dss_dl = DataLoader(datasets._dss[ds_key], batch_size=bs, collate_fn=lambda x: x, shuffle=False, num_workers = n_threads)

            cvs_dls = [
                    DataLoader(_tds[mk], batch_size=bs, collate_fn=lambda x: x, shuffle=False, num_workers = n_threads) for mk in _mtc 
                    ]
    
            with torch.no_grad():
                # the first data in the tuple should be the dataset
                # to next ones are from cvs_dls, ordered according to _mtc
                for data in tqdm(zip(dss_dl, *cvs_dls), disable=not verbose, total=ceil(n_samples/bs)):
                    ds_data = data[0]
                    cvs_data = {mk: data[i+1] for i, mk in enumerate(_mtc)}

                    # inferece
                    model(ds_data[input_key].to(device))
                    for mk in _mtc:
                        act_data = activations_parser(model._acts)
                        cvs_data[mk][mk] = reducers[mk](act_data=act_data[mk])
    
            # save as stacked
            self._corevds[ds_key] = _ModuleWiseStack(tds=_tds)

        # reset the model to NOT get activations
        model.set_activations(save_input=False, save_output=False)
    
        return

    def normalize_corevectors(self, **kwargs):
        '''
        Normalize corevectors.

        Args:
        - wrt (str): selects which loader to compute the means and stds, the other loaders are normalized using this loader's means and stds. Defaults to None. 
        - batch_size (int): Creates dataloader to do computation in batch size. Defaults to 64.
        - n_threads (int): 'num_workers' passed to 'torch.utils.data.DataLoader'. Defaults to 1.
        - verbose (bool): print progress messages.
        '''
        self.check_uncontexted()

        wrt = kwargs['wrt']
        bs = kwargs.get('batch_size', 64) 
        n_threads = kwargs.get('n_threads', 1) 
        verbose = kwargs.get('verbose', False) 

        if self._corevds == {}:
            raise RuntimeError('No corevectors to normalize. Run get_corevectors() first.')

        if verbose: print(f'\n---- Applying normalization w.r.t. {wrt}\n')
        
        
        
        # denormalize
        for ds_key in self._corevds.keys():
            if ds_key not in self._normalizations: self._normalizations[ds_key] = {}

            for mk in self._corevds[ds_key].keys():
                # get old normalization
                _denorm = False
                norm_file = Path(self._corevds_files[ds_key][mk].as_posix()+'.normalization') 
                if norm_file.exists():
                    if verbose: print(f'Found normalization file {norm_file}. Data will be denormalized and renormalized.')
                    _denorm = True
                    old_means, old_stds = torch.load(norm_file, weights_only=True)

                dl = DataLoader(self._corevds[ds_key].tds[mk], batch_size=bs, collate_fn=lambda x: x, num_workers=n_threads)

                if _denorm:
                    if verbose: print(f'Deormalizing {ds_key}, {mk}')
                    for data in tqdm(dl, disable=not verbose, total=len(dl)):
                        data[mk] = data[mk]*old_stds + old_means
                
        # renormalize
        means = self._corevds[wrt].mean(dim=0)
        stds = self._corevds[wrt].std(dim=0)
        for ds_key in self._corevds.keys():
            for mk in self._corevds[ds_key].keys():
                if verbose: print(f'Normalizing {ds_key}, {mk}')

                norm_file = Path(self._corevds_files[ds_key][mk].as_posix()+'.normalization') 
                dl = DataLoader(self._corevds[ds_key].tds[mk], batch_size=bs, collate_fn=lambda x: x, num_workers=n_threads)
                for data in tqdm(dl, disable=not verbose, total=len(dl)):
                    data[mk] = (data[mk] - means[mk])/stds[mk]

                torch.save((means[mk], stds[mk]), norm_file)
                self._normalizations[ds_key][mk] = means, stds

        return
    
    def load_only(self, **kwargs):
        '''
        Load already computed corevectors.

        Args:
        - loaders (list[str]): load the specified loaders
        - names dict(str:str): Dictionary with key being the module name, and value being a name to append to the PTD file with the corevectors. Corevectors are loaded from a file with name `<loader>/<key>.<name>. If `None` is given as `name` loads  `<loader>/<key>`.
        - mode (str): Opens the file with the specified mode. See 'tensordict.PersistentTensorDict.from_h5()' for details. Defaults to 'r'.
        - norm_file (str): load the normalization information. Defaults to None. 
        - verbose (bool): print progress messages.
        '''
        self.check_uncontexted()

        loaders = kwargs['loaders']
        names = kwargs['names']
        mode = kwargs.get('mode', 'r')
        verbose = kwargs.get('verbose', False)

        self.__close()

        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting data from {ds_key}\n')
            
            _tds = {}
            self._normalizations[ds_key] = {}
            self._corevds_files[ds_key] = {}
            for mk in names.keys():
                if names[mk] == None:
                    file_path = self.path/ds_key/mk
                else:
                    file_path = self.path/ds_key/(mk+'.'+names[mk])
                self._corevds_files[ds_key][mk] = file_path

                if verbose: print(f'Loading file {file_path}. ')
                _td = PersistentTensorDict.from_h5(file_path, mode=mode)
                _tds[mk] = _td

                norm_file = Path(file_path.as_posix()+'.normalization') 
                if norm_file.exists():
                    if verbose: print(f'Loading normalization from {norm_file}. ')
                    self._normalizations[ds_key][mk] = torch.load(norm_file, weights_only=True)

            self._corevds[ds_key] = _ModuleWiseStack(tds=_tds)
            if verbose: print('Loaded n_samples: ', len(self._corevds[ds_key]))
        return
    
    def __close(self):
        if self._corevds != {}:
            for ds_key in self._corevds:
                self._corevds[ds_key].close()

        # reset these
        self._corevds = {}
        self._corevds_files = {}
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
