# General python stuff
from tqdm import tqdm
from functools import partial
from math import ceil

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

# Our stuff
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.datasets.dataset_base import DatasetBase 
from peepholelib.coreVectors.parsers import from_dataset
from peepholelib.coreVectors.prediction_fns import multilabel_classification
    
def get_activations(self, **kwargs):
    self.check_uncontexted()
    
    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
    n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 1 
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    model = self._model
    num_classes = self._model.num_classes
    device = self._model.device 
    hooks = model.get_hooks()
    
    assert(isinstance(model, ModelWrap))
    if self._dss == {}:
        raise RuntimeError('No dataset parsed. Run `parse_ds()` first.')
    
    for ds_key in self._dss:
        if verbose: print(f'\n ---- Getting data from {ds_key}\n')
       
        #------------------------------------------------
        # pre-allocate activations
        #------------------------------------------------

        # check if in and out activations exist
        if model._si:
            if not ('in_activations' in self._dss[ds_key]):
                if verbose: print('adding in act tensorDict')
                self._dss[ds_key]['in_activations'] = TensorDict(batch_size=n_samples)
            else: 
                if verbose: print('In activations exist.')
        if 'in_activations' in self._dss[ds_key]: self._dss[ds_key]['in_activations'].batch_size = torch.Size((n_samples,)) 

        if model._so:
            if not ('out_activations' in self._dss[ds_key]):
                if verbose: print('adding out act tensorDict')
                self._dss[ds_key]['out_activations'] = TensorDict(batch_size=n_samples)
            else: 
                if verbose: print('Out activations exist.')
        if 'out_activations' in self._dss[ds_key]: self._dss[ds_key]['out_activations'].batch_size = torch.Size((n_samples,)) 
        
        # dry run to get shapes
        img = self._dss[ds_key][0:1]['image'].to(device)
        print('img shape: ', img.shape)
        model(img)

        # check if module exists in in_ and out_activations
        _modules_to_save = []
        for mk in model.get_target_modules():
            # prevents double entries 
            _lts = None

            # allocate for input activations 
            if model._si and (not (mk in self._dss[ds_key]['in_activations'])):
                if verbose: print('allocating in act module: ', mk)
                # Seems like when loading from memory the batch size gets overwritten with all dims, so we over-overwrite it.
                act_shape = hooks[mk].in_shape
                self._dss[ds_key]['in_activations'][mk] = MMT.empty(shape=torch.Size((n_samples,)+act_shape))
                _lts = mk

            # allocate for output activations 
            if model._so and (not (mk in self._dss[ds_key]['out_activations'])):
                if verbose: print('allocating out act module: ', mk)
                act_shape = hooks[mk].out_shape
                self._dss[ds_key]['out_activations'][mk] = MMT.empty(shape=torch.Size((n_samples,)+act_shape))
                _lts = mk
            
            if _lts != None: _modules_to_save.append(_lts)
        
        if verbose: print('modules to save: ', _modules_to_save)
        if len(_modules_to_save) == 0:
            if verbose: print(f'No new activations for {ds_key}, skipping')
            continue

        # ---------------------------------------
        # get activations
        # ---------------------------------------
        
        # create a temp dataloader to iterate over images
        ds_dl = DataLoader(self._dss[ds_key], batch_size=bs, collate_fn = lambda x: x, shuffle=False, num_workers=n_threads) 
        
        if verbose: print('Computing activations')
        for data in tqdm(ds_dl, disable=not verbose):
            with torch.no_grad():
                model(ds_data['image'].to(device))
            
            for mk in _modules_to_save:
                if model._si:
                    data['in_activations'][mk] = hooks[mk].in_activations[:].cpu()

                if model._so:
                    data['out_activations'][mk] = hooks[mk].out_activations[:].cpu()
    return 

