# General python stuff
from tqdm import tqdm
from functools import partial
from math import ceil

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

from .parsers import from_dataset
from .prediction_fns import multilabel_classification

def get_in_activations(x):
    return x['in_activations']

def get_out_activations(x):
    return x['out_activations']

def get_coreVectors(self, **kwargs):
    self.check_uncontexted()
    
    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
    n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 1 
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    model = self._model 
    device = self._model.device 
    hooks = model.get_hooks()

    reduction_fns = kwargs['reduction_fns'] if 'reduction_fns' in kwargs else lambda x, y:x[y]
    activations_parser = kwargs['activations_parser'] if 'activations_parser' in kwargs else get_in_activations

    if reduction_fns.keys() != model._target_modules.keys(): 
        raise RuntimeError(f'Keys inconsistency between reduction_fns and target_modules \n reduction_fns keys: {reduction_fns.keys()} \n target_modules: {model._target_modules.keys()}')
    
    self._corevds = {}
    for ds_key in self._dss:
        #------------------------------------------------
        # pre-allocate corevectors
        #------------------------------------------------

        if verbose: print(f'\n ---- Getting core vectors for {ds_key}\n')
        file_path = self.path/(self.name+'.'+ds_key)
        self._cvs_file_paths[ds_key] = file_path

        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._corevds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
            n_samples = len(self._corevds[ds_key])
            if verbose: print('loaded n_samples: ', n_samples)
            self._corevds[ds_key].batch_size = torch.Size((n_samples,)) 
        else:
            n_samples = self._n_samples[ds_key]
            self._corevds[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
            if verbose: print('loader n_samples: ', n_samples) 

        if verbose: print(f'\n ---- Getting core vectors for {ds_key}\n')
        
        # check if module in and out activations exist
        _modules_to_save = []

        # allocate for core vectors 
        for mk in model._target_modules.keys(): 
            if not (mk in self._corevds[ds_key]):
                if verbose: print('allocating core vectors for module: ', mk)

                if not self._actds == None:
                    # get cv shape from data
                    _a0 = activations_parser(self._actds[ds_key][0])[mk]
                    _act0 = _a0.reshape((1,)+_a0.shape)
                else:
                    # Dry run to get shape
                    with torch.no_grad():
                        model(self._dss[ds_key][0:1]['image'].to(device))
                        
                    if model._si:
                        _act0 = hooks[mk].in_activations[:] 
                    elif model._so:
                        _act0 = hooks[mk].out_activations[:] 
                    
                cv_shape = reduction_fns[mk](act_data=_act0).shape[1:]

                self._corevds[ds_key][mk] = MMT.empty(shape=((n_samples,)+cv_shape))
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
        if not self._actds == None:
            if verbose: print('Using saved activation')

            act_dl = DataLoader(self._actds[ds_key], batch_size=bs, collate_fn=activations_parser, shuffle=False, num_workers = n_threads)

            for cvs_data, act_data in tqdm(zip(cvs_dl, act_dl), disable=not verbose, total=len(cvs_dl)):
                for mk in _modules_to_save:
                    cvs_data[mk] = reduction_fns[mk](act_data=act_data)
            
        else:
            if verbose: print('Getting activations from model')

            ds_dl = DataLoader(self._dss[ds_key], batch_size=bs, collate_fn=lambda x: x, shuffle=False, num_workers = n_threads)

            for cvs_data, ds_data in tqdm(zip(cvs_dl, ds_dl), disable=not verbose, total=len(cvs_dl)):
                with torch.no_grad():
                    model(ds_data['image'].to(device))
                    
                for mk in _modules_to_save:
                    if model._si:
                        act_data = hooks[mk].in_activations[:] 
                    elif model._so:
                        act_data = hooks[mk].out_activations[:] 

                    cvs_data[mk] = reduction_fns[mk](act_data=act_data)

    return
              
