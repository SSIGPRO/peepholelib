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
    save_input = kwargs['save_input'] if 'save_input' in kwargs else True
    save_output = kwargs['save_output'] if 'save_output' in kwargs else False 

    model = self._model
    device = self._model.device 
    
    assert(isinstance(model, ModelWrap))
    if self._dss == {}:
        raise RuntimeError('No dataset parsed. Run `parse_ds()` first.')
    
    # set the model to get activations
    model.set_activations(save_input=save_input, save_output=save_output)

    for ds_key in self._dss:
        if verbose: print(f'\n ---- Getting data from {ds_key}\n')
        n_samples = len(self._dss[ds_key])
       
        #------------------------------------------------
        # pre-allocate activations
        #------------------------------------------------

        # dry run to get shapes
        with torch.no_grad():
            model(self._dss[ds_key]['image'][0:1].to(device))

        _modules_to_save = {}
        for act_key in model._acts:
            if not (act_key in self._dss[ds_key]):
                if verbose: print(f'adding {act_key} tensorDict')
                self._dss[ds_key][act_key] = TensorDict(batch_size=n_samples)
            else:
                # this is a workaound a pesky bug when loading PTDs with thensors on it, which sets bs = data.shape instead of data.shape[0]
                self._dss[ds_key][act_key].batch_size = torch.Size((n_samples,)) 
            
            _modules_to_save[act_key] = []
            for mk in model._acts[act_key]:
                # allocate activations 
                if not (mk in self._dss[ds_key][act_key]):
                    if verbose: print(f'allocating {act_key} for {mk}')
                    act_shape = model._acts[act_key][mk].shape[1:]
                    self._dss[ds_key][act_key][mk] = MMT.empty(shape=torch.Size((n_samples,)+act_shape))
                    _modules_to_save[act_key].append(mk)

        if verbose: print('modules to save: ', _modules_to_save)
        # check if there is anything to save for ds_key
        skip = True
        for ms in _modules_to_save.values():
            if len(ms) != 0:
               skip = False 
        
        if skip:
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
                model(data['image'].to(device))
            
            for act_key in _modules_to_save:
                for mk in _modules_to_save[act_key]:
                    data[act_key][mk] = model._acts[act_key][mk].cpu()

    # reset the model to NOT get activations
    model.set_activations(save_input=False, save_output=False)

    return 

