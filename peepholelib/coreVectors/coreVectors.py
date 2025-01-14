# torch stuff
import torch
from torch.utils.data import DataLoader 
from tensordict import TensorDict, PersistentTensorDict

# generic python stuff
from pathlib import Path
from tqdm import tqdm
from functools import partial

class CoreVectors():
    from .dataset import get_coreVec_dataset
    from .activations import get_activations
    from .svd_coreVectors import get_coreVectors

    def __init__(self, **kwargs):
        self.path = Path(kwargs['path'])
        self.name = Path(kwargs['name'])
        # create folder
        self.path.mkdir(parents=True, exist_ok=True)

        self._model = kwargs['model'] if 'model' in kwargs else None  

        # computed in get_coreVec_dataset()
        self._cvs_file_paths = {} 
        self._n_samples = {} 
        self._corevds = {} # filled in get_coreVectors()

        # computed in get_activations()
        self._act_file_paths = {} 
        self._actds = {}
        
        # set in normalize_corevectors() 
        self._norm_mean = None 
        self._norm_std = None 
        
        # Set on __enter__() and __exit__()
        # read before each function
        self._is_contexted = False

        # computed in get_dataloaders()
        self._loaders = {}
        return
     
    def normalize_corevectors(self, **kwargs):
        self.check_uncontexted()

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False 
        bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64 
        from_file = Path(kwargs['from_file']) if 'from_file' in kwargs else None
        wrt = kwargs['wrt'] if 'wrt' in kwargs else None
        to_file = Path(kwargs['to_file']) if 'to_file' in kwargs  else None
        target_layers = kwargs['target_layers'] if 'target_layers' in kwargs  else None

        if wrt == None and from_file == None:
            raise RuntimeError(f'Specify `wrt` or `from_file`.')
        
        if from_file != None:
            if verbose: print(f'Loading normalization from {from_file}')
            means, stds = torch.load(from_file)
        else: # wrt will not be None
            if verbose: print(f'Computing normalization from {wrt}')
            means = self._corevds[wrt]['coreVectors'].mean(dim=0)
            stds = self._corevds[wrt]['coreVectors'].std(dim=0)
            
        if target_layers != None:
            keys_to_pop = tuple(means.keys()-target_layers)

            for k in keys_to_pop:
                means.pop(k,default=None)
                stds.pop(k,default=None)
            
        for ds_key in self._corevds:
            if verbose: print(f'\n ---- Normalizing core vectors for {ds_key}\n')
            dl = DataLoader(self._corevds[ds_key], batch_size=bs, collate_fn=lambda x: x)
            
            for batch in tqdm(dl, disable=not verbose, total=len(dl)):
                batch['coreVectors'] = (batch['coreVectors'] - means)/stds
        
        if to_file != None:
            to_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save((means, stds), to_file)

        self._norm_mean = means
        self._norm_std = stds

        return

    def get_dataloaders(self, **kwargs):
        self.check_uncontexted()
        
        _bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
        if isinstance(_bs, int):
            batch_dict = {key: _bs for key in self._corevds}
        elif isinstance(_bs, dict):
            batch_dict = _bs
        else:
            raise RuntimeError('Batch size should be a dict or an integer')

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False 
        if self._loaders:
            if verbose: print('Loaders exist. Returning existing ones.')
            return self._loaders

        # Create dataloader for each corevecs TensorDicts 
        _loaders = {}
        for ds_key in self._corevds:
            if verbose: print('creating dataloader for: ', ds_key)
            _loaders[ds_key] = DataLoader(
                    dataset = self._corevds[ds_key],
                    batch_size = batch_dict[ds_key], 
                    collate_fn = lambda x: x
                    )

        self._loaders = _loaders 
        return self._loaders
    
    def load_only(self, **kwargs):
        self.check_uncontexted()

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        loaders = kwargs['loaders']
        norm_file = Path(kwargs['norm_file']) if 'norm_file' in kwargs else None 

        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting data from {ds_key}\n')
            file_path = self.path/(self.name.name+'.'+ds_key)
            self._cvs_file_paths[ds_key] = file_path
            
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._corevds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r')

            self._n_samples[ds_key] = len(self._corevds[ds_key])
            if verbose: print('loaded n_samples: ', self._n_samples[ds_key])
       
        if norm_file != None:
            if verbose: print('Loading normalization info.')
            means, stds = torch.load(norm_file_path)
            self._norm_mean = means 
            self._norm_std = stds

        return
    
    def __enter__(self):
        self._is_contexted = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        verbose = True 

        if self._corevds == None:
            if verbose: print('no corevds to close.')
        else:
            for ds_key in self._corevds:
                if verbose: print(f'closing {ds_key}')
                self._corevds[ds_key].close()
            
        if self._actds == None:
            if verbose: print('no actds to close.')
        else:
            for ds_key in self._actds:
                if verbose: print(f'closing {ds_key}')
                self._actds[ds_key].close()

        self._is_contexted = False 
        return

    def check_uncontexted(self):
        if not self._is_contexted:
            raise RuntimeError('Function should be called within context manager')
        return
