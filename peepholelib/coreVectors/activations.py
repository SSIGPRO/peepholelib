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
from .parsers import from_dataset
from .prediction_fns import multilabel_classification
    
def get_activations(self, **kwargs):
    self.check_uncontexted()
    
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    datasets = kwargs['datasets']
    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
    n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 1 

    key_list = kwargs['key_list'] if 'key_list' in kwargs else ['image', 'label']
    ds_parser = kwargs['ds_parser'] if 'ds_parser' in kwargs else from_dataset 
    pred_fn = kwargs['pred_fn'] if 'pred_fn' in kwargs else multilabel_classification

    model = self._model
    num_classes = self._model.num_classes
    device = self._model.device 
    hooks = model.get_hooks()

    for ds_key in datasets:
        if verbose: print(f'\n ---- Getting data from {ds_key}\n')
        file_path = self.path/('activations.'+ds_key)
        self._act_file_paths[ds_key] = file_path     

        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._actds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            self._n_samples[ds_key] = len(self._actds[ds_key])
            
            n_samples = self._n_samples[ds_key]
            if verbose: print('loaded n_samples: ', n_samples)
        else:
            self._n_samples[ds_key] = len(datasets[ds_key])
            n_samples = self._n_samples[ds_key] 
            if verbose: print('created persistent tensor dict with n_samples: ', n_samples)
            self._actds[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')

            #------------------------
            # Pre-allocation 
            #------------------------
            if verbose: print('Allocating images and labels')
            _data = [datasets[ds_key][0]]
            data = ds_parser(_data, key_list=key_list)

            for key in key_list:
                _d = data[key][0]
                # pre-allocation activations
                if _d.shape == torch.Size([]):
                    self._actds[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,))) 
                else:
                    self._actds[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,)+_d.shape))
           
            # Close PTD create with mode 'w' and re-open it with mode 'r+'
            # This is done so we can use multiple workers for reading and writting
            self._actds[ds_key].close()
            self._actds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            #------------------------
            # copy images and labels
            #------------------------
            # create dataloader of input dataset and activations
            dl_ds = DataLoader(dataset=datasets[ds_key], batch_size=bs, collate_fn=partial(ds_parser, key_list=key_list), shuffle=False) 
            dl_act = DataLoader(self._actds[ds_key], batch_size=bs, collate_fn=lambda x:x, shuffle=False, num_workers=n_threads)

            if verbose: print('Copying images and labels')
            for data_in, data_t in tqdm(zip(dl_ds, dl_act), disable=not verbose, total=ceil(n_samples/bs)): 
                for key in key_list:
                    data_t[key] = data_in[key]
            
        #------------------------------------------------
        # pre-allocate predictions, results, activations
        #------------------------------------------------
        act_td = self._actds[ds_key]

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
        
        # check if module exists in in_ and out_activations
        _modules_to_save = []
        for mk in model.get_target_modules():
            # prevents double entries 
            _lts = None

            # allocate for input activations 
            if model._si and (not (mk in act_td['in_activations'])):
                if verbose: print('allocating in act module: ', mk)
                # Seems like when loading from memory the batch size gets overwritten with all dims, so we over-overwrite it.
                act_shape = hooks[mk].in_shape
                act_td['in_activations'][mk] = MMT.empty(shape=torch.Size((n_samples,)+act_shape))
                _lts = mk

            # allocate for output activations 
            if model._so and (not (mk in act_td['out_activations'])):
                if verbose: print('allocating out act module: ', mk)
                act_shape = hooks[mk].out_shape
                act_td['out_activations'][mk] = MMT.empty(shape=torch.Size((n_samples,)+act_shape))
                _lts = mk
            
            if _lts != None: _modules_to_save.append(_lts)
        
        if verbose: print('modules to save: ', _modules_to_save)
        if len(_modules_to_save) == 0:
            if verbose: print(f'No new activations for {ds_key}, skipping')
            continue
        
        # to check if pred and results data exist 
        has_pred = 'pred' in act_td 
        
        # allocate memory for pred and result
        if not has_pred:
            act_td['output'] = MMT.empty(shape=torch.Size((n_samples,num_classes)))
            act_td['pred'] = MMT.empty(shape=torch.Size((n_samples,)))
            act_td['result'] = MMT.empty(shape=torch.Size((n_samples,)))
        
        # ---------------------------------------
        # compute predictions and get activations
        # ---------------------------------------
        
        # create a temp dataloader to iterate over images
        act_dl = DataLoader(act_td, batch_size=bs, collate_fn = lambda x: x, shuffle=False, num_workers=n_threads) 
        
        if verbose: print('Computing activations')
        for act_data in tqdm(act_dl, disable=not verbose):
            with torch.no_grad():
                y_predicted = model(act_data['image'].to(device))
            
            # do not save predictions and results if it is already there
            if not has_pred:
                predicted_labels = pred_fn(y_predicted)
                act_data['output'] = y_predicted
                act_data['pred'] = predicted_labels
                act_data['result'] = predicted_labels == act_data['label']
            
            for mk in _modules_to_save:
                if model._si:
                    act_data['in_activations'][mk] = hooks[mk].in_activations[:].cpu()

                if model._so:
                    act_data['out_activations'][mk] = hooks[mk].out_activations[:].cpu()
    return 

