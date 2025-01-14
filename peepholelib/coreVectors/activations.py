# General python stuff
from tqdm import tqdm

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

def get_activations(self, **kwargs):
    self.check_uncontexted()
    model = self._model
    device = self._model.device 
    hooks = model.get_hooks()

    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False 

    if not self._corevds:
        raise RuntimeError('No data found. Please run get_coreVec_dataset() first.')

    for ds_key in self._corevds:
        if verbose: print(f'\n ---- Getting activations for {ds_key}\n')
        
        file_path = self.path/(self.name.name+'.activations.'+ds_key)
        n_samples = self._n_samples[ds_key]       

        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._actds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
            if verbose: print(f'loaded {file_path}')
        else:
            if verbose: print(f'creating {file_path}') 
            self._actds[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')

        #------------------------------------------------
        # pre-allocate predictions, results, activations
        #------------------------------------------------
        act_td = self._actds[ds_key]
        cvs_td = self._corevds[ds_key]

        # check if in and out activations exist
        if model._si and (not ('in_activations' in act_td)):
            if verbose: print('adding in act tensorDict')
            act_td['in_activations'] = TensorDict(batch_size=n_samples)
        elif verbose: print('In activations exist.')
        if 'in_activations' in act_td: act_td['in_activations'].batch_size = torch.Size((n_samples,)) 

        if model._so and (not ('out_activations' in act_td)):
            if verbose: print('adding out act tensorDict')
            act_td['out_activations'] = TensorDict(batch_size=n_samples)
        elif verbose: print('Out activations exist.')
        if 'out_activations' in act_td: act_td['out_activations'].batch_size = torch.Size((n_samples,)) 
        
        # check if layer exists in in_ and out_activations
        _layers_to_save = []
        for lk in model.get_target_layers():
            # prevents double entries 
            _lts = None

            # allocate for input activations 
            if model._si and (not (lk in act_td['in_activations'])):
                if verbose: print('allocating in act layer: ', lk)
                # Seems like when loading from memory the batch size gets overwritten with all dims, so we over-overwrite it.
                act_shape = hooks[lk].in_shape
                act_td['in_activations'][lk] = MMT.empty(shape=torch.Size((n_samples,)+act_shape))
                _lts = lk

            # allocate for output activations 
            if model._so and (not (lk in act_td['out_activations'])):
                if verbose: print('allocating out act layer: ', lk)
                act_shape = hooks[lk].out_shape
                act_td['out_activations'][lk] = MMT.empty(shape=torch.Size((n_samples,)+act_shape))
                _lts = lk
             
            if _lts != None: _layers_to_save.append(_lts)
        
        if verbose: print('Layers to save: ', _layers_to_save)
        if len(_layers_to_save) == 0:
            if verbose: print(f'No new activations for {ds_key}, skipping')
            continue
        
        # to check if pred and results data exist 
        has_pred = 'pred' in cvs_td 
        
        # allocate memory for pred and result
        if not has_pred:
            cvs_td['pred'] = MMT.empty(shape=torch.Size((n_samples,)))
            cvs_td['result'] = MMT.empty(shape=torch.Size((n_samples,)))
        
        # ---------------------------------------
        # compute predictions and get activations
        # ---------------------------------------
        
        # create a temp dataloader to iterate over images
        cvs_dl = DataLoader(cvs_td, batch_size=bs, collate_fn = lambda x: x, shuffle=False) 
        act_dl = DataLoader(act_td, batch_size=bs, collate_fn = lambda x: x, shuffle=False) 
        
        if verbose: print('Computing activations')
        
        for cvs_data, act_data in tqdm(zip(cvs_dl, act_dl), disable=not verbose, total=len(cvs_dl)):
            with torch.no_grad():
                y_predicted = model(cvs_data['image'].to(device))
            
            # do not save predictions and results if it is already there
            if not has_pred:
                predicted_labels = y_predicted.argmax(axis = 1).cpu()
                cvs_data['pred'] = predicted_labels
                cvs_data['result'] = predicted_labels == cvs_data['label']
            
            for lk in _layers_to_save:
                if model._si:
                    act_data['in_activations'][lk] = hooks[lk].in_activations[:].cpu()

                if model._so:
                    act_data['out_activations'][lk] = hooks[lk].out_activations[:].cpu()
    return 

