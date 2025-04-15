# python stuff
from pathlib import Path
from tqdm import tqdm
from math import ceil

# Stuff used in evaluation ... will get out from here
from collections import Counter
import numpy as np

# plotting stuff
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

from peepholelib.peepholes import drill_base as driller

class Peepholes:
    def __init__(self, **kwargs):
        self.target_modules = kwargs['target_modules'] # list of peep modules
        self.path = Path(kwargs['path'])
        self.name = kwargs['name']
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        # (dict) one classifier per module
        self._driller = kwargs['driller'] 

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
        - corevectors (peepholelib.CoreVectors): corevectors object containing corevectors and activations
        - batchsize (int): batchsize to process corevectors into peepholes
        '''
        self.check_uncontexted()

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        cvs = kwargs['corevectors'] 
        bs = kwargs['batch_size']

        for (ds_key, cvds), ( _, actds) in zip(cvs._corevds.items(), cvs._actds.items()):
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
                    self._phs[ds_key][module]['peepholes'] = MMT.empty(shape=(n_samples, self._driller[module].nl_model))
                    modules_to_compute.append(module)
                else:
                    if verbose: print(f'Peepholes for {module} already present. Skipping.')
                    
            #------------------------ 
            # computing peepholes
            #------------------------
            # create dataloaders
            dl_phs = DataLoader(self._phs[ds_key], batch_size=bs, collate_fn=lambda x:x)
            dl_cvs = DataLoader(cvds, batch_size=bs, collate_fn=lambda x: x)
            dl_act = DataLoader(actds, batch_size=bs, collate_fn=lambda x: x)
            if verbose: print(f'\n ---- computing peepholes for modules {modules_to_compute}\n')
            for cvs, acts, phs in tqdm(zip(dl_cvs, dl_act, dl_phs), disable=not verbose, total=ceil(n_samples/bs)):
                for module in modules_to_compute:
                    phs[module]['peepholes'] = self._driller[module](cvs=cvs, acts=acts)

        return 

    def get_scores(self, **kwargs):
        '''
        Compute scores (score_max and score_entropy) from precomputed peepholes.
        '''
        self.check_uncontexted()
        
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 32
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
    
    def evaluate_dists(self, **kwargs):
        self.check_uncontexted()
        
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False 
        acts = kwargs['activations']
        score_type = kwargs['score_type']
        bins = kwargs['bins'] if 'bins' in kwargs else 100

        for module in self.target_modules:
            print(f'\n-------------\nEvaluating Distributions for module {module}\n-------------\n') 
            
            n_dss = len(self._phs.keys())
            fig, axs = plt.subplots(1, n_dss+1, sharex='all', sharey='all', figsize=(4*(1+n_dss), 4))
            
            m_ok, s_ok, m_ko, s_ko = {}, {}, {}, {}

            for i, ds_key in enumerate(self._phs.keys()):       # train val test
                if verbose: print(f'Evaluating {ds_key}')
                results = acts[ds_key]['result']
                scores = self._phs[ds_key][module]['score_'+score_type]
                oks = (scores[results == True]).detach().cpu().numpy()
                kos = (scores[results == False]).detach().cpu().numpy()

                m_ok[ds_key], s_ok[ds_key] = oks.mean(), oks.std()
                m_ko[ds_key], s_ko[ds_key] = kos.mean(), kos.std()


                #--------------- 
                # plotting
                #---------------
                ax = axs[i+1]
                sb.histplot(data=pd.DataFrame({'score': oks}), ax=ax, bins=bins, x='score', stat='density', label='ok n=%d'%len(oks), alpha=0.5)
                sb.histplot(data=pd.DataFrame({'score': kos}), ax=ax, bins=bins, x='score', stat='density', label='ko n=%d'%len(kos), alpha=0.5)
                ax.set_xlabel('score: '+score_type)
                ax.set_ylabel('%')
                ax.title.set_text(ds_key)
                ax.legend(title='dist')
            
            # plot train and test distributions
            ax = axs[0]
            scores = self._phs['train'][module]['score_'+score_type].detach().cpu().numpy()
            sb.histplot(data=pd.DataFrame({'score': scores}), ax=ax, bins=bins, x='score', stat='density', label='train n=%d'%len(scores), alpha=0.5)
            scores = self._phs['val'][module]['score_'+score_type].detach().cpu().numpy()
            sb.histplot(data=pd.DataFrame({'score': scores}), ax=ax, bins=bins, x='score', stat='density', label='val n=%d'%len(scores), alpha=0.5)
            ax.set_ylabel('%')
            ax.set_xlabel('score: '+score_type)
            ax.legend(title='datasets')
            plt.savefig((self.path/self.name).as_posix()+f'.{module}.png', dpi=300, bbox_inches='tight')
            plt.close()

            if verbose: print('oks mean, std, n: ', m_ok, s_ok, len(oks), '\nkos, mean, std, n', m_ko, s_ko, len(kos))

        return m_ok, s_ok, m_ko, s_ko

    def evaluate(self, **kwargs): 
        self.check_uncontexted()

        cvs = kwargs['coreVectors']
        score_type = kwargs['score_type']
        
        for module in self.target_modules:
            quantiles = torch.arange(0, 1, 0.001) # setting quantiles list
            prob_train = self._phs['train'][module]['peepholes']
            prob_val = self._phs['val'][module]['peepholes']
            
            # TODO: vectorize
            conf_t = self._phs['train'][module]['score_'+score_type].detach().cpu() 
            conf_v = self._phs['val'][module]['score_'+score_type].detach().cpu() 
 
            th = [] 
            lt = []
            lf = []

            c = cvs['val'].dataset['result'].detach().cpu().numpy()
            cntt = Counter(c) 
            
            for q in quantiles:
                perc = torch.quantile(conf_t, q)
                th.append(perc)
                idx = torch.argwhere(conf_v > perc)[:,0]

                # TODO: vectorize
                cnt = Counter(c[idx]) 
                lt.append(cnt[True]/cntt[True]) 
                lf.append(cnt[False]/cntt[False])

            plt.figure()
            x = quantiles.numpy()
            y1 = np.array(lt)
            y2 = np.array(lf)
            plt.plot(x, y1, label='OK', c='b')
            plt.plot(x, y2, label='KO', c='r')
            plt.plot(np.array([0., 1.]), np.array([1., 0.]), c='k')
            plt.legend()
            plt.savefig((self.path/self.name).as_posix()+f'.{module}.png', dpi=300, bbox_inches='tight')
            plt.close()

        return np.linalg.norm(y1-y2), np.linalg.norm(y1-y2)

    def get_dataloaders(self, **kwargs):
        self.check_uncontexted()

        batch_dict = kwargs['batch_dict'] if 'batch_dict' in kwargs else {key: 64 for key in self._phs}
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False 

        if self._loaders:
            if verbose: print('Loaders exist. Returning existing ones.')
            return self._loaders

        _loaders = {}
        for key in self._phs:
            if verbose: print('creating dataloader for: ', key)
            _loaders[key] = DataLoader(
                    dataset = self._phs[key],
                    batch_size = batch_dict[key], 
                    collate_fn = lambda x: x
                    )

        self._loaders = _loaders 
        return self._loaders

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
