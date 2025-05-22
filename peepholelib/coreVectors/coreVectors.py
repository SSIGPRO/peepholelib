# torch stuff
import torch
from torch.utils.data import DataLoader 
from tensordict import TensorDict, PersistentTensorDict

# generic python stuff
from pathlib import Path
from tqdm import tqdm

class CoreVectors():
    from .parse_ds import parse_ds 
    from .activations import get_activations
    from .get_coreVectors import get_coreVectors

    def __init__(self, **kwargs):
        self.path = Path(kwargs['path'])
        self.name = kwargs['name']
        
        # create folder
        self.path.mkdir(parents=True, exist_ok=True)

        self._model = kwargs['model'] if 'model' in kwargs else None  
        # computed in parse_ds() and get_activations()
        self._dss = None 

        # computed in get_coreVectors()
        self._corevds = None 

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
        n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 1 

        from_file = Path(kwargs['from_file']) if 'from_file' in kwargs else None
        wrt = kwargs['wrt'] if 'wrt' in kwargs else None
        to_file = Path(kwargs['to_file']) if 'to_file' in kwargs  else None
        target_layers = kwargs['target_layers'] if 'target_layers' in kwargs else None
        
        if self._corevds == None:
            raise RuntimeError('No corevectors to normalize. Run get_corevectors() first.')

        if wrt == None and from_file == None:
            raise RuntimeError(f'Specify `wrt` or `from_file`.')
        
        if from_file != None:
            if verbose: print(f'Loading normalization from {from_file}')
            means, stds = torch.load(from_file, weights_only=False)
        else: # wrt will not be None
            if verbose: print(f'Computing normalization from {wrt}')
            means = self._corevds[wrt].mean(dim=0)
            
            stds = self._corevds[wrt].std(dim=0)

        if target_layers != None:
            keys_to_pop = tuple(means.keys()-target_layers)
            for k in keys_to_pop:
                means.pop(k, default=None)
                stds.pop(k, default=None)

        for ds_key in self._corevds:
            if verbose: print(f'\n ---- Normalizing core vectors for {ds_key}\n')
            dl = DataLoader(self._corevds[ds_key], batch_size=bs, collate_fn=lambda x: x, num_workers = n_threads)
            
            for batch in tqdm(dl, disable=not verbose, total=len(dl)):
                for _k in means.keys():
                    batch[_k] = (batch[_k]- means[_k])/stds[_k]

        if to_file != None:
            to_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save((means, stds), to_file)

        self._norm_mean = means
        self._norm_std = stds

        return

    def load_only(self, **kwargs):
        self.check_uncontexted()

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        loaders = kwargs['loaders']
        norm_file = Path(kwargs['norm_file']) if 'norm_file' in kwargs else None 

        self._corevds = {}
        self._dss = {}
        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting data from {ds_key}\n')
            
            _cvs_file_paths = self.path/(self.name+'.'+ds_key)
            _dss_file_paths = self.path/('dss.'+ds_key)

            if verbose: print(f'Loading files {self._cvs_file_paths[ds_key]} and {self._dss_file_paths[ds_key]} from disk. ')
            self._corevds[ds_key] = PersistentTensorDict.from_h5(_cvs_file_paths, mode='r')
            self._dss[ds_key] = PersistentTensorDict.from_h5(_dss_file_paths, mode='r')

            _n_samples = len(self._corevds[ds_key])
            if verbose: print('loaded n_samples: ', _n_samples)
       
        if norm_file != None:
            if verbose: print('Loading normalization info.')
            self._norm_mean, self._norm_std = torch.load(norm_file)

        return
    
    def __enter__(self):
        self._is_contexted = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        verbose = True 

        if self._dss == None:
            if verbose: print('no dss to close.')
        else:
            for ds_key in self._dss:
                if verbose: print(f'closing {ds_key}')
                self._dss[ds_key].close()

        if self._corevds == None:
            if verbose: print('no corevds to close.')
        else:
            for ds_key in self._corevds:
                if verbose: print(f'closing {ds_key}')
                self._corevds[ds_key].close()

        self._is_contexted = False 
        return

    def check_uncontexted(self):
        if not self._is_contexted:
            raise RuntimeError('Function should be called within context manager')
        return
