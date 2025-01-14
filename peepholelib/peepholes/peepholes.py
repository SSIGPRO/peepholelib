# python stuff
from pathlib import Path
from tqdm import tqdm
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

class Peepholes:
    def __init__(self, **kwargs):
        self.layer = kwargs['layer']
        self.path = Path(kwargs['path'])
        self.name = Path(kwargs['name'])
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        # create folder
        self.path.mkdir(parents=True, exist_ok=True)

        self._classifier = kwargs['classifier'] 

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
        - dataloader (DataLoader): Dataloader containing data to be parsed with the paser function set on __init__() 
        '''
        self.check_uncontexted()

        if self._classifier._empp == None:
            raise RuntimeError('No prediction probabilities. Please run classifier.compute_empirical_posteriors() first.')
        _empp = self._classifier._empp.to(self.device)
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        dls = kwargs['loaders']

        layer = self.layer 

        for ds_key in dls:
            bs = dls[ds_key].batch_size
            if verbose: print(f'\n ---- Getting peepholes for {ds_key}\n')
            file_path = self.path/(self.name.name+'.'+ds_key)
           
            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
                n_samples = len(self._phs[ds_key])
                if verbose: print('loaded n_samples: ', n_samples)
            else:
                n_samples = len(dls[ds_key].dataset)
                if verbose: print('loader n_samples: ', n_samples) 
                self._phs[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
            
            #-----------------------------------------
            # Pre-allocate peepholes
            #-----------------------------------------
            if not layer in self._phs[ds_key]:
                if verbose: print('allocating peepholes for layer: ', layer)
                self._phs[ds_key][layer] = TensorDict(batch_size=n_samples)
                self._phs[ds_key][layer]['peepholes'] = MMT.empty(shape=(n_samples, self._classifier.nl_model))
             
                #----------------------------------------- 
                # computing peepholes
                #-----------------------------------------
                if verbose: print('\n ---- computing peepholes \n')
                dl_t = DataLoader(self._phs[ds_key], batch_size=bs, collate_fn=lambda x:x)
                for batch in tqdm(zip(dls[ds_key], dl_t), disable=not verbose, total=len(dl_t)):
                    data_in, data_t = batch
                    cp = self._classifier.classifier_probabilities(batch=data_in, verbose=verbose).to(self.device)
                    lp = cp@_empp
                    lp /= lp.sum(dim=1, keepdim=True)
                    data_t[layer]['peepholes'] = lp.cpu()
            else:
                if verbose: print('Peepholes for {layer} already present. Skipping.')
        return 

    def get_scores(self, **kwargs):
        '''
        Compute scores (score_max and score_entropy) from precomputed peepholes.
        '''
        self.check_uncontexted()
        
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 32
        bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64 
        layer = self.layer 

        if self._phs == None:
            raise RuntimeError('No core vectors present. Please run get_peepholes() first.')

        for ds_key in self._phs:
            if verbose: print(f'\n ---- Getting scores for {ds_key}\n')
            file_path = self.path / (self.name.name + '.' + ds_key)
    
            #-----------------------------------------
            # Check if peepholes exist before computing scores
            #-----------------------------------------
            n_samples = len(self._phs[ds_key])

            if layer not in self._phs[ds_key]:
                raise ValueError(f"Peepholes for layer {layer} do not exist. Please run get_peepholes() first.")
            
            if 'peepholes' not in self._phs[ds_key][layer]:
                raise ValueError(f"Peepholes do not exist in layer {layer}. Please run get_peepholes() first.")
            
            #-----------------------------------------
            # Check if scores already exist
            #-----------------------------------------
            if 'score_max' in self._phs[ds_key][layer] and 'score_entropy' in self._phs[ds_key][layer]:
                if verbose: print(f"Scores already computed for layer {layer}. Skipping computation.")
                continue 

            #-----------------------------------------
            # Pre-allocate scores
            #-----------------------------------------
            if verbose: print('Allocating scores for layer:', layer)
            self._phs[ds_key][layer].batch_size = torch.Size((n_samples,))
            self._phs[ds_key][layer]['score_max'] = MMT.empty(shape=(n_samples,))
    
            self._phs[ds_key][layer]['score_entropy'] = MMT.empty(shape=(n_samples,))
             
            #-----------------------------------------
            # Compute scores
            #-----------------------------------------
            if verbose: print('\n ---- Computing scores \n')
            _dl = DataLoader(self._phs[ds_key], batch_size=bs, collate_fn=lambda x: x)
            for batch in tqdm(_dl, disable=not verbose, total=len(_dl)):
                peepholes = batch[layer]['peepholes']
                batch[layer]['score_max'] = torch.max(peepholes, dim=1).values
                batch[layer]['score_entropy'] = torch.sum(peepholes * torch.log(peepholes + 1e-12), dim=1)
    
        return
    
    def load_only(self, **kwargs):
        self.check_uncontexted()

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        loaders = kwargs['loaders']

        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting peepholes for {ds_key}\n')
            file_path = self.path/(self.name.name+'.'+ds_key)
           
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r')
        
        return
    
    def evaluate_dists(self, **kwargs):
        self.check_uncontexted()
         
        layer = self.layer 
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False 
        cv_dls = kwargs['coreVectors']
        score_type = kwargs['score_type']
        bins = kwargs['bins'] if 'bins' in kwargs else 100

        print('\n-------------\nEvaluating Distributions\n-------------\n') 
        
        n_dss = len(self._phs.keys())
        fig, axs = plt.subplots(1, n_dss+1, sharex='all', sharey='all', figsize=(4*(1+n_dss), 4))
        
        m_ok, s_ok, m_ko, s_ko = {}, {}, {}, {}

        for i, ds_key in enumerate(self._phs.keys()):
            if verbose: print('Evaluating {ds_key}')
            results = cv_dls[ds_key].dataset['result']
            scores = self._phs[ds_key][layer]['score_'+score_type]
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
        scores = self._phs['train'][layer]['score_'+score_type].detach().cpu().numpy()
        sb.histplot(data=pd.DataFrame({'score': scores}), ax=ax, bins=bins, x='score', stat='density', label='train n=%d'%len(scores), alpha=0.5)
        scores = self._phs['val'][layer]['score_'+score_type].detach().cpu().numpy()
        sb.histplot(data=pd.DataFrame({'score': scores}), ax=ax, bins=bins, x='score', stat='density', label='val n=%d'%len(scores), alpha=0.5)
        ax.set_ylabel('%')
        ax.set_xlabel('score: '+score_type)
        ax.legend(title='datasets')

        plt.savefig((self.path/self.name).as_posix()+'.png', dpi=300, bbox_inches='tight')
        plt.close()

        if verbose: print('oks mean, std, n: ', m_ok, s_ok, len(oks), '\nkos, mean, std, n', m_ko, s_ko, len(kos))
        return m_ok, s_ok, m_ko, s_ko

    def evaluate(self, **kwargs): 
        self.check_uncontexted()

        layer = self.layer 
        cvs = kwargs['coreVectors']
        score_type = kwargs['score_type']

        quantiles = torch.arange(0, 1, 0.001) # setting quantiles list
        prob_train = self._phs['train'][layer]['peepholes']
        prob_val = self._phs['val'][layer]['peepholes']
        
        # TODO: vectorize
        conf_t = self._phs['train'][layer]['score_'+score_type].detach().cpu() 
        conf_v = self._phs['val'][layer]['score_'+score_type].detach().cpu() 
 
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
        plt.savefig((self.path/self.name).as_posix()+'.png')
        plt.close()
        
        # TODO: make dist to diagonal
        # TODO: save and load evaluation scores
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
