# General python stuff
from tqdm import tqdm
from functools import partial
from math import ceil

# torch stuff
import torch
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

# our stuff
from peepholelib.models.model_wrap import get_in_activations

def get_coreVectors(self, **kwargs):
    '''
    Compute and save corevectos. Corevectors are saved directly on disk using a 'tensordict.PersistentTensorDict' at 'self.path/self.name.<loader>', with 'loader' being the loader keys (see peepholelib.datasets).
    Pre-allocation is done with shapes obtained via a dry-run. Checks are performed for existing loaders and existing modules, which are skipped.
    If activations are present in 'datasets._dss', use the saved values, but saving activations is memory heavy. Otherwise, pass the input images through the model in batches and get the activations directly from the model (see 'peepholelib.model_wrap').

    Args:
    - datasets (dict(str: peepholelib.datasets.parsedDataset.ParsedDataset)): Parsed datasets.
    - loaders (list[str]): List of loaders in `datasets.keys()` to compute corevectors. If `None` uses `datasets._dss.keys()`. Defaults to `None`.
    - input_key TODO
    - reducers (dict(str: Callable)): A dictionary with keys being the module names as per the model's state_dict, and values being a callable foo(acts) which takes as input the model's batched activations and returns a dimentionality reduced version of its outputs. 
    - activations_parser (callable): A function for parsing activations. Defaults to 'get_in_activations()' (see peepholelib.models.model_wrap.py for details on how we get the activations).
    - batch_size (int): Creates dataloader to do computation in batch size. Defaults to 64.
    - n_threads (int): 'num_workers' passed to 'torch.utils.data.DataLoader'. Defaults to 1.
    - verbose (bool): print progress messages.
    '''
    self.check_uncontexted()
    
    datasets = kwargs.get('datasets')
    loaders = kwargs.get('loaders', None)
    input_key = kwargs.get('input_key','image')
    reducers = kwargs.get('reducers')
    activations_parser = kwargs.get('activations_parser', get_in_activations)
    bs = kwargs.get('batch_size', 64) 
    n_threads = kwargs.get('n_threads', 1) 
    save_input = kwargs.get('save_input', True)
    save_output = kwargs.get('save_output', False) 
    verbose = kwargs.get('verbose', False) 

    model = self._model 
    device = self._model.device 

    if reducers.keys() != model._target_modules.keys(): 
        raise RuntimeError(f'Keys inconsistency between reducers and target_modules \n reducers keys: {reducers.keys()} \n target_modules: {model._target_modules.keys()}')

    # set the model to get activations
    model.set_activations(save_input=save_input, save_output=save_output)

    self._corevds = {}
    for ds_key in datasets._dss:
        #------------------------------------------------
        # pre-allocate corevectors
        #------------------------------------------------
        if verbose: print(f'\n ---- Getting core vectors for {ds_key}\n')
        file_path = self.path/(self.name+'.'+ds_key)

        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._corevds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
            n_samples = len(self._corevds[ds_key])
            if verbose: print('loaded n_samples: ', n_samples)
            # this is a workaround for when loading PTDs with already populated MMTs
            self._corevds[ds_key].batch_size = torch.Size((n_samples,)) 
        else:
            n_samples = len(datasets._dss[ds_key])
            self._corevds[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
            if verbose: print('created corevectors with n_samples: ', n_samples) 

            if verbose: print(f'\n ---- Getting core vectors for {ds_key}\n')
        
        # check if module in and out activations exist
        _modules_to_save = []

        # sample for dry run
        with torch.no_grad():
            model(datasets._dss[ds_key][0:1][input_key].to(device))
            _act0 = activations_parser(model._acts)

        for mk in model._target_modules.keys(): 
            if not (mk in self._corevds[ds_key]):
                # Dry run to get CV shape
                _cv = reducers[mk](act_data=_act0[mk])
                cv_shape = _cv.shape[1:]

                # allocate for core vectors 
                if verbose: print('allocating core vectors for module: ', mk)
                self._corevds[ds_key][mk] = MMT.empty(shape=((n_samples,)+cv_shape), dtype=_cv.dtype)
                _modules_to_save.append(mk)

        if verbose: print('Modules to save: ', _modules_to_save)
        
        # Close PTD create with mode 'w' and re-open it with mode 'r+'
        # This is done so we can use multiple workers for reading and writting
        self._corevds[ds_key].close()
        self._corevds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

        # ---------------------------------------
        # compute corevectors 
        # ---------------------------------------
        if len(_modules_to_save) == 0:
            print(f'No new core vectors for {ds_key}, skipping')
            continue
        if verbose: print(f'\n ---- Getting corevectors for {ds_key}\n')

        cvs_dl = DataLoader(self._corevds[ds_key], batch_size=bs, collate_fn=lambda x: x, shuffle=False, num_workers = n_threads) 

        if verbose: print('Getting activations from model')

        ds_dl = DataLoader(datasets._dss[ds_key], batch_size=bs, collate_fn=lambda x: x, shuffle=False, num_workers = n_threads)

        for cvs_data, ds_data in tqdm(zip(cvs_dl, ds_dl), disable=not verbose, total=len(cvs_dl)):
            with torch.no_grad():
                model(ds_data[input_key].to(device))
                
            for mk in _modules_to_save:
                act_data = activations_parser(model._acts)
                cvs_data[mk] = reducers[mk](act_data=act_data[mk])

    # reset the model to NOT get activations
    model.set_activations(save_input=False, save_output=False)

    return
