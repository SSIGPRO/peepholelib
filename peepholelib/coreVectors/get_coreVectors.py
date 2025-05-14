# General python stuff
from tqdm import tqdm

# torch stuff
import torch
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

def get_in_activations(x):
    return x['in_activations']

def get_coreVectors(self, **kwargs):
    self.check_uncontexted()
    
    model = self._model 
    device = self._model.device 
    normalize_wrt = kwargs['normalize_wrt'] if 'normalize_wrt' in kwargs else None 
    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
    n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 1 

    reduction_fns = kwargs['reduction_fns'] if 'reduction_fns' in kwargs else lambda x, y:x[y]
    activations_parser = kwargs['activations_parser'] if 'activations_parser' in kwargs else get_in_activations 

    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    if not self._actds:
        raise RuntimeError('No activations found. Please run get_activations() first.')
    
    if reduction_fns.keys() != model._target_modules.keys(): 
        raise RuntimeError(f'Keys inconsistency between reduction_fns and target_modules \n reduction_fns keys: {reduction_fns.keys()} \n target_modules: {model._target_modules.keys()}')
        
    for ds_key in self._actds:

        if verbose: print(f'\n ---- Getting core vectors for {ds_key}\n')
        file_path = self.path/(self.name.name+'.'+ds_key)
        self._cvs_file_paths[ds_key] = file_path

        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._corevds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
            n_samples = len(self._corevds[ds_key])
            if verbose: print('loaded n_samples: ', n_samples)
            self._corevds[ds_key].batch_size = torch.Size((n_samples,)) 
        else:
            n_samples = len(self._actds[ds_key])
            self._corevds[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
            if verbose: print('loader n_samples: ', n_samples) 

        if verbose: print(f'\n ---- Getting core vectors for {ds_key}\n')
        
        # check if module in and out activations exist
        _modules_to_save = []
        
        # allocate for core vectors 
        for mk in model._target_modules.keys(): 
            if not (mk in self._corevds[ds_key]):
                if verbose: print('allocating core vectors for module: ', mk)
                # get cv shape from data
                _a0 = activations_parser(self._actds[ds_key][0])[mk]
                _act0 = _a0.reshape((1,)+_a0.shape)
                cv_shape = reduction_fns[mk](act_data=_act0).shape[1:]

                self._corevds[ds_key][mk] = MMT.empty(shape=((n_samples,)+cv_shape))
                _modules_to_save.append(mk)

        if verbose: print('modules to save: ', _modules_to_save)
        if len(_modules_to_save) == 0:
            print(f'No new core vectors for {ds_key}, skipping')
            continue
        
        # Close PTD create with mode 'w' and re-open it with mode 'r+'
        # This is done so we can use multiple workers for reading and writting
        self._corevds[ds_key].close()
        self._corevds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

        cvs_td = self._corevds[ds_key]
        act_td = self._actds[ds_key]

        # ---------------------------------------
        # compute corevectors 
        # ---------------------------------------

        # create a temp dataloader to iterate over images
        cvs_dl = DataLoader(cvs_td, batch_size=bs, collate_fn = lambda x: x, shuffle=False, num_workers = n_threads) 
        act_dl = DataLoader(act_td, batch_size=bs, collate_fn = activations_parser, shuffle=False, num_workers = n_threads)

        if verbose: print('Computing core vectors')
        
        if verbose: print(f'\n ---- Getting corevectors for {ds_key}\n')
        for cvs_data, act_data in tqdm(zip(cvs_dl, act_dl), disable=not verbose, total=len(cvs_dl)):
            for mk in _modules_to_save:
                cvs_data[mk] = reduction_fns[mk](act_data=act_data[mk])

    return        