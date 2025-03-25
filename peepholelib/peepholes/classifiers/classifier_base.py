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

        # computed in fit()
        self._classifier = None

        # set in fit()
        self._cvs = None 

        # computer in compute_empirical_posteriors()
        self._empp = None

        # defined in save() or load()
        self._empp_file = None
        self._clas_path = None
        return
    
    @abc.abstractmethod
    def load(self, **kwargs):
        self._empp_file = self.path/(self._suffix+'.empp.pt')
        self._empp = torch.load(self._empp_file)
        pass 

    @abc.abstractmethod
    def save(self, **kwargs):
        self._empp_file = self.path/(self._suffix+'.empp.pt')
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
        - verbose (Bool): print some stuff
        '''
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        actds = kwargs['actds']
        corevds = kwargs['corevds']
        bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
        # pre-allocate empirical posteriors
        _empp = torch.zeros(self.nl_class, self.nl_model)
        
        # create dataloaders
        acts_dl = DataLoader(actds, batch_size=bs, collate_fn=lambda x: x)
        cvs_dl = DataLoader(corevds, batch_size=bs, collate_fn=lambda x: x)

        # iterate over _fit_data
        if verbose: print('Computing empirical posterior')
        for act, cvs in tqdm(zip(acts_dl, cvs_dl), disable=not verbose):
            data, label = self.parser(act=act, cvs=cvs, **self.parser_kwargs)
            data, label = data.to(self.device), label.to(self.device)
            preds = self._classifier.predict(data)
            
            for p, l in zip(preds, label):
                _empp[int(p), int(l)] += 1
       
        # normalize to get empirical posteriors
        _empp /= _empp.sum(dim=1, keepdim=True)

        # replace NaN with 0
        _empp = torch.nan_to_num(_empp)
        self._empp = _empp
        
        return 
    
    def __call__(self, **kwargs):
        '''
        Compute the peephole base on the empirical posterior 
        '''
        cvs = kwargs['cvs']
        verbose = kwargs['verbose']

        # # check for empiracal posterios `_empp`
        if self._empp == None:
            raise RuntimeError('No prediction probabilities. Please run classifiers[layer].compute_empirical_posteriors() first.')
        _empp = self._empp.to(self.device)
        data = self.parser(cvs=cvs, **self.parser_kwargs)
        cp = self.classifier_probabilities(data=data, verbose=verbose).to(self.device)
        lp = cp@_empp
        lp /= lp.sum(dim=1, keepdim=True)

        return lp
