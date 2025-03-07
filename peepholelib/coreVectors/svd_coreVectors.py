# General python stuff
from tqdm import tqdm

# torch stuff
import torch
import torchvision
from torchvision.models.vision_transformer import VisionTransformer
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.functional import pad

def get_coreVectors(self, **kwargs):
    self.check_uncontexted()
    
    model = self._model 
    device = self._model.device 
    normalize_wrt = kwargs['normalize_wrt'] if 'normalize_wrt' in kwargs else None 
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64

    reduction_fns = kwargs['reduction_fns'] if 'reduction_fns' in kwargs else lambda x, y:x[y]
    shapes = kwargs['shapes'] # TODO dry run if not specified the shape
     
    if not self._corevds:
        raise RuntimeError('No dataset data found. Please run get_coreVec_dataset() first.')
    
    if not self._actds:
        raise RuntimeError('No activations found. Please run get_activations() first.')
        
    if reduction_fns.keys() != shapes.keys(): 
        raise RuntimeError(f'Keys inconsistency between reduction_fns and shapes \n reduction_fns keys: {reduction_fns.keys()} \n shapes keys: {shapes.keys()}')

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
        
        # TODO If we don't have shapes you must iterate here for lk in model.get_target_layers():
        
        for lk, corev_size in shapes.items(): 

            # allocate for core vectors 
            if not (lk in cvs_td['coreVectors']):
                if verbose: print('allocating core vectors for layer: ', lk)
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
            
            if verbose: print(f'\n ---- Getting corevectors for {lk}\n')
            
            for cvs_data, act_data in tqdm(zip(cvs_dl, act_dl), disable=not verbose, total=len(cvs_dl)):
                
                cvs_data['coreVectors'][lk] = reduction_fns[lk](*act_data['in_activations'][lk])


    return        
