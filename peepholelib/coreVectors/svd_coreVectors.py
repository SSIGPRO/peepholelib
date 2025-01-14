# General python stuff
from tqdm import tqdm

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.functional import pad

def reduct_matrices_from_svds(svd, lk):
    return svd[lk]['Vh']

def get_coreVectors(self, **kwargs):
    self.check_uncontexted()
    
    model = self._model 
    device = self._model.device 
    normalize_wrt = kwargs['normalize_wrt'] if 'normalize_wrt' in kwargs else None 
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
    reduct_matrices = kwargs['reduct_matrices']

    parser = kwargs['parser'] if 'parser' in kwargs else lambda x, y:x[y]
    parser_kwargs = kwargs['parser_kwargs'] if 'parser_kwargs' in kwargs and 'parser' in kwargs else dict()
     
    if not self._corevds:
        raise RuntimeError('No dataset data found. Please run get_coreVec_dataset() first.')
    
    if not self._actds:
        raise RuntimeError('No activations found. Please run get_activations() first.')

    for ds_key in self._corevds:
        if verbose: print(f'\n ---- Getting core vectors for {ds_key}\n')
        n_samples = self._n_samples[ds_key]       
        
        cvs_td = self._corevds[ds_key]
        act_td = self._actds[ds_key]

        # create corevectors TensorDict if needed 
        if not 'coreVectors' in cvs_td:
            if verbose: print('adding core vectors tensorDict')
            cvs_td['coreVectors'] = TensorDict(batch_size=n_samples)
        elif verbose: print('core vectors TensorDict exists.')
        cvs_td['coreVectors'].batch_size = torch.Size((n_samples,))

        # check if layer in and out activations exist
        _layers_to_save = []
        for lk in model.get_target_layers():
            reduct_m = parser(reduct_matrices, lk, **parser_kwargs)

            # allocate for core vectors 
            if not (lk in cvs_td['coreVectors']):
                if verbose: print('allocating core vectors for layer: ', lk)
                corev_size = reduct_m.shape[0] 
                cvs_td['coreVectors'][lk] = MMT.empty(shape=torch.Size((n_samples,)+(corev_size,)))
                _layers_to_save.append(lk)

        if verbose: print('Layers to save: ', _layers_to_save)
        if len(_layers_to_save) == 0:
            print(f'No new core vectors for {ds_key}, skipping')
            continue

        # ---------------------------------------
        # compute corevectors 
        # ---------------------------------------

        # create a temp dataloader to iterate over images
        cvs_dl = DataLoader(cvs_td, batch_size=bs, collate_fn = lambda x: x, shuffle=False) 
        act_dl = DataLoader(act_td, batch_size=bs, collate_fn = lambda x: x, shuffle=False) 
        
        if verbose: print('Computing core vectors')
        for lk in _layers_to_save:
            reduct_m = parser(reduct_matrices, lk, **parser_kwargs).to(device)
            if verbose: print(f'\n ---- Getting corevectors for {lk}\n')
            
            layer = model._target_layers[lk]
            if isinstance(layer, torch.nn.Linear):
                for cvs_data, act_data in tqdm(zip(cvs_dl, act_dl), disable=not verbose, total=len(cvs_dl)):
                    n_act = act_data['in_activations'][lk].shape[0]
                    acts = act_data['in_activations'][lk]
                    acts_flat = acts.flatten(start_dim=1)
                    ones = torch.ones(n_act, 1)
                    _acts = torch.hstack((acts_flat, ones)).to(device)
                    phs = (reduct_m@_acts.T).T
                    cvs_data['coreVectors'][lk] = phs.cpu()
            if isinstance(layer, torch.nn.Conv2d):
                pad_mode = layer.padding_mode if layer.padding_mode != 'zeros' else 'constant'
                padding = _reverse_repeat_tuple(layer.padding, 2) 
                for cvs_data, act_data in tqdm(zip(cvs_dl, act_dl), disable=not verbose, total=len(cvs_dl)):
                    n_act = act_data['in_activations'][lk].shape[0]
                    acts = act_data['in_activations'][lk]
                    acts_pad = pad(acts, pad=padding, mode=pad_mode)

                    acts_flat = acts_pad.flatten(start_dim=1)
                    ones = torch.ones(n_act, 1)
                    _acts = torch.hstack((acts_flat, ones)).to(device)
                    phs = (reduct_m@_acts.T).T
                    cvs_data['coreVectors'][lk] = phs.cpu()

    return        
