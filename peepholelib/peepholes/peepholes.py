# python stuff
from pathlib import Path
from tqdm import tqdm
from math import ceil

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

from peepholelib.peepholes import drill_base as driller

class Peepholes:
    def __init__(self, **kwargs):
        self.path = Path(kwargs['path'])
        self.name = kwargs['name']
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        # Set in get_peepholes() 
        self.target_modules = None # list of peep modules
        self._drillers = None 

        # computed in get_peepholes
        self._phs = {} 
        
        # computed in get_dataloaders()
        self._loaders = None

        # Set on __enter__() and __exit__()
        # read before each function
        self._is_contexted = False

        return

    def get_peepholes(self, **kwargs):
        '''
        Compute model probabilities from classifier probabilities and empirical posteriors.
        
        Args:
        - verbose (bool): print progress messages
        - corevectors (peepholelib.CoreVectors): corevectors object containing corevectors and datasets 
        - batchsize (int): batchsize to process corevectors into peepholes
        '''
        self.check_uncontexted()

        self._drillers = kwargs['drillers'] 
        self.target_modules = kwargs['target_modules'] # list of peep modules

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        cvs = kwargs['corevectors'] 
        bs = kwargs['batch_size']
        n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 1 

        for (ds_key, cvds), ( _, dssds) in zip(cvs._corevds.items(), cvs._dss.items()):
            if verbose: print(f'\n ---- Getting peepholes for {ds_key}\n')
            file_path = self.path/(self.name+'.'+ds_key)
            
            # create/load PersistentTensorDict file
            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
                n_samples = len(self._phs[ds_key])
                if verbose: print('loaded n_samples: ', n_samples)
            else:
                n_samples = len(cvds)
                if verbose: print('loader n_samples: ', n_samples) 
                self.path.mkdir(parents=True, exist_ok=True)
                self._phs[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
            
            modules_to_compute = []
            for module in self.target_modules:
                if not module in self._phs[ds_key]:
                    #------------------------
                    # Pre-allocate peepholes
                    #------------------------
                    if verbose: print('allocating peepholes for module: ', module)
                    self._phs[ds_key][module] = TensorDict(batch_size=n_samples)
                    self._phs[ds_key][module]['peepholes'] = MMT.empty(shape=(n_samples, self._drillers[module].nl_model))
                    modules_to_compute.append(module)
                else:
                    if verbose: print(f'Peepholes for {module} already present. Skipping.')

            # Close PTD create with mode 'w' and re-open it with mode 'r+'
            # This is done so we can use multiple workers for reading and writting
            self._phs[ds_key].close()
            self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            #------------------------ 
            # computing peepholes
            #------------------------
            # create dataloaders
            dl_phs = DataLoader(self._phs[ds_key], batch_size=bs, collate_fn=lambda x:x, num_workers = n_threads)
            dl_cvs = DataLoader(cvds, batch_size=bs, collate_fn=lambda x: x, num_workers = n_threads)
            dl_dss = DataLoader(dssds, batch_size=bs, collate_fn=lambda x: x, num_workers = n_threads)

            if len(modules_to_compute) == 0:
                if verbose: print('No modules to compute for {ds_key}. Skipping.')
                continue

            if verbose: print(f'\n ---- computing peepholes for modules {modules_to_compute}\n')
            for cvs, dss, phs in tqdm(zip(dl_cvs, dl_dss, dl_phs), disable=not verbose, total=ceil(n_samples/bs)):
                for module in modules_to_compute:
                    phs[module]['peepholes'] = self._drillers[module](cvs=cvs, dss=dss)

        return 

    def get_scores(self, **kwargs):
        '''
        Compute scores (score_max and score_entropy) from precomputed peepholes.
        '''
        self.check_uncontexted()
        
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 1 
        bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64 

        if self._phs == None:
            raise RuntimeError('No core vectors present. Please run get_peepholes() first.')

        for ds_key in self._phs:
            if verbose: print(f'\n ---- Getting scores for {ds_key}\n')
            file_path = self.path / (self.name + '.' + ds_key)
    
            #-----------------------------------------
            # Check if peepholes exist before computing scores
            #-----------------------------------------
            n_samples = len(self._phs[ds_key])

            for module in self.target_modules:
                if module not in self._phs[ds_key]:
                    raise ValueError(f"Peepholes for module {module} do not exist. Please run get_peepholes() first.")
                
                if 'peepholes' not in self._phs[ds_key][module]:
                    raise ValueError(f"Peepholes do not exist in module {module}. Please run get_peepholes() first.")
                    
                #-----------------------------------------
                # Check if scores already exist
                #-----------------------------------------
                if 'score_max' in self._phs[ds_key][module] and 'score_entropy' in self._phs[ds_key][module]:
                    if verbose: print(f"Scores already computed for module {module}. Skipping computation.")
                    continue 

                #-----------------------------------------
                # Pre-allocate scores
                #-----------------------------------------
                if verbose: print('Allocating scores for module:', module)
                self._phs[ds_key][module].batch_size = torch.Size((n_samples,))
                self._phs[ds_key][module]['score_max'] = MMT.empty(shape=(n_samples,))
                self._phs[ds_key][module]['score_entropy'] = MMT.empty(shape=(n_samples,))
                
                #-----------------------------------------
                # Compute scores
                #-----------------------------------------
                if verbose: print('\n ---- Computing scores \n')
                _dl = DataLoader(self._phs[ds_key], batch_size=bs, collate_fn=lambda x: x)
                for batch in tqdm(_dl, disable=not verbose, total=len(_dl)):
                    peepholes = batch[module]['peepholes']
                    batch[module]['score_max'] = torch.max(peepholes, dim=1).values
                    batch[module]['score_entropy'] = torch.sum(peepholes * torch.log(peepholes + 1e-12), dim=1)
    
        return
    
    def load_only(self, **kwargs):
        '''
        Load the peepholes 
        '''
        self.check_uncontexted()

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        loaders = kwargs['loaders']

        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting peepholes for {ds_key}\n')
            file_path = self.path/(self.name+'.'+ds_key)
           
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r')

        return
    

    def get_conceptograms(self, **kwargs):
        '''
        Get conceptograms from peepholes
        '''
        self.check_uncontexted()

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False

        if self._phs == None:
            raise RuntimeError('No core vectors present. Please run get_peepholes() first.')

        self._conceptograms = {}
        for ds_key in self._phs:
            if verbose: print(f'\n ---- Getting conceptograms for {ds_key}\n')
            file_path = self.path / (self.name + '.' + ds_key)

            #-----------------------------------------
            # Check if peepholes exist before computing scores
            #-----------------------------------------
            n_samples = len(self._phs[ds_key])

            for module in self.target_modules:
                if module not in self._phs[ds_key]:
                    raise ValueError(f"Peepholes for module {module} do not exist. Please run get_peepholes() first.")

                if 'peepholes' not in self._phs[ds_key][module]:
                    raise ValueError(f"Peepholes do not exist in module {module}. Please run get_peepholes() first.")

            self._conceptograms[ds_key] = torch.stack([self._phs[ds_key][layer]['peepholes'] for layer in self.target_modules],dim=1)

        return

    def __enter__(self):
        self._is_contexted = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        verbose = True 

        if self._phs == None:
            if verbose: print('no peepholes to close. doing nothing.')
            return

        for ds_key in self._phs:
            if verbose: print(f'closing {ds_key}')
            self._phs[ds_key].close()
            
        self._is_contexted = False 
        return

    def check_uncontexted(self):
        if not self._is_contexted:
            raise RuntimeError('Function should be called within context manager')
        return
