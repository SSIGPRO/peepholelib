# python stuff
import abc  
from pathlib import Path
from tqdm import tqdm

# torch stuff
import torch
from torch.utils.data import DataLoader
from peepholelib.peepholes.drill_base import DrillBase

def null_parser(**kwargs):
    data = kwargs['data']
    return data['data'], data['label'] 
    
class ClassifierBase(DrillBase): 
    def __init__(self, **kwargs):
        DrillBase.__init__(self, **kwargs)

        # number of classes in classifier a.k.a. number of clusters
        self.nl_class = kwargs['nl_classifier'] if 'nl_classifier' in kwargs else None# computed in fit()

        self._classifier = None

        # set in fit()
        self._cvs = None 

        # computer in compute_empirical_posteriors()
        self._empp = None

        # defined in __init__(), used in save() and load()
        self._clas_path = None
        self._empp_file = None
        
        # used in save() or load()
        self._suffix += f'.nl_class={self.nl_class}'

        return
    
    @abc.abstractmethod
    def load(self, **kwargs):
        self._empp = torch.load(self._empp_file).to(self.device)
        pass 

    @abc.abstractmethod
    def save(self, **kwargs):
        torch.save(self._empp, self._empp_file)
        pass

    @abc.abstractmethod
    def fit(self, **kwargs):
        pass        

    @abc.abstractmethod
    def classifier_probabilities(self, **kwargs):
        pass
    
    def compute_empirical_posteriors(self, **kwargs):
        '''
        Compute the empirical posterior matrix P, where P(g, c) is the probability that a sample assigned to classifier's class g belongs to the model's class c.

        Args:
        - corevectors (peepholelib.coreVectors.CoreVectors): Corevectors.
        - loader (str): Which loader used for computing the Empirical Posteriors, usually 'train'. Defaults to 'train'. 
        - batch_size: Do the computation in batchs. Defaults to 64.
        - verbose (Bool): Print progress messages. 
        '''

        cvs = kwargs.get('corevectors')
        loader = kwargs.get('loader', 'train')
        bs = kwargs.get('batch_size', 64)
        verbose = kwargs.get('verbose', False)

        # pre-allocate empirical posteriors
        _empp = torch.zeros(self.nl_class, self.nl_model)
        
        # create dataloaders
        dss_dl = DataLoader(cvs._dss[loader], batch_size=bs, collate_fn=lambda x: x, shuffle=False)
        cvs_dl = DataLoader(cvs._corevds[loader], batch_size=bs, collate_fn=lambda x: x, shuffle=False)

        # iterate over _fit_data
        if verbose: print('Computing empirical posterior')
        for _dss, _cvs in tqdm(zip(dss_dl, cvs_dl), disable=not verbose):
            data, label = self.parser(cvs=_cvs, dss=_dss)
            data, label = data.to(self.device), label.to(self.device)
            preds = self._classifier.predict(data)
            for p, l in zip(preds, label):
                _empp[int(p), int(l)] += 1

        # normalize to get empirical posteriors
        _empp /= _empp.sum(dim=1, keepdim=True)

        # replace NaN with 0
        _empp = torch.nan_to_num(_empp)


        self._empp = _empp.to(self.device)
        
        return 
    
    def __call__(self, **kwargs):
        '''
        Compute the peephole base on the empirical posterior. 
        
        Args:
        - cvs (torch.tensor): Batch of corevectors, will be parsed with self.parser (see __init__()).
        - verbose (bool): Print progress messages.
        
        '''
        cvs = kwargs['cvs']
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False 

        # # check for empiracal posterios `_empp`
        if self._empp == None:
            raise RuntimeError('No prediction probabilities. Please run classifiers[layer].compute_empirical_posteriors() first.')
        data = self.parser(cvs=cvs)
        cp = self.classifier_probabilities(data=data, verbose=verbose).to(self.device)
        lp = cp@self._empp
        lp /= lp.sum(dim=1, keepdim=True)

        return lp
