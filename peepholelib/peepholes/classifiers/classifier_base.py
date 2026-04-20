# python stuff
import abc  
from pathlib import Path
from tqdm import tqdm

# torch stuff
import torch
from torch.utils.data import DataLoader
from peepholelib.peepholes.drill_base import DrillBase

class ClassifierBase(DrillBase, metaclass=abc.ABCMeta): 
    def __init__(self, **kwargs):
        DrillBase.__init__(self, **kwargs)

        # number of classes in classifier a.k.a. number of clusters
        self.nl_class = kwargs['nl_classifier'] if 'nl_classifier' in kwargs else None# computed in fit()
        self.label_key = kwargs.get('label_key', 'label')
        self.reducer = kwargs['reducer']

        self.parser = self.reducer.parser 
        # computed in inheriting classes 
        self._classifier = None

        # computer in compute_empirical_posteriors()
        self._empp = None

        # defined in __init__(), used in save() and load()
        self._clas_path = None
        self._empp_file = None

        return
    
    @abc.abstractmethod
    def load(self, **kwargs):
        if self._empp_file.exists():
            self._empp = torch.load(self._empp_file).to(self.device)
        return 

    @abc.abstractmethod
    def save(self, **kwargs):
        if self._empp != None:
            torch.save(self._empp, self._empp_file)
        return 

    @abc.abstractmethod
    def predict(self, data):
        pass

    @abc.abstractmethod
    def fit(self, **kwargs):
        pass        

    @abc.abstractmethod
    def classifier_probabilities(self, **kwargs):
        pass
    
    def _compute_empirical_posteriors(self, **kwargs):
        '''
        Compute the empirical posterior matrix P, where P(g, c) is the probability that a sample assigned to classifier's class g belongs to the model's class c.

        Args:
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): Parsed datasets respective the `coreVectors`.
        - corevectors (peepholelib.coreVectors.coreVectors.CoreVectors): Corevectors respective the `datasets`.
        - loader (str): Which loader used for computing the Empirical Posteriors, usually 'train'. Defaults to 'train'. 
        - batch_size: Do the computation in batchs. Defaults to 512.
        - verbose (Bool): Print progress messages. 
        - label_key (str): key to get labels from
        '''
        
        dss = kwargs['datasets']
        cvs = kwargs['corevectors']
        loader = kwargs.get('loader', 'train')

        # pre-allocate empirical posteriors
        _empp = torch.zeros(self.nl_class, self.nl_model, device=self.device)

        data = self.parser(cvs=cvs._corevds[loader][self.target_module])
        label = dss._dss[loader][self.label_key].to(self.device)
        preds = self.predict(data).to(self.device)
        indices = preds.long() * self.nl_model + label.long()
        _empp = torch.bincount(indices, minlength=self.nl_class * self.nl_model).reshape(self.nl_class, self.nl_model).float()

        # normalize to get empirical posteriors
        _empp /= _empp.sum(dim=1, keepdim=True)

        # replace NaN with 0
        self._empp = torch.nan_to_num(_empp).to(self.device)
        
        return 
    
    def __call__(self, **kwargs):
        '''
        Compute the peephole base on the empirical posterior. 
        
        Args:
        - cvs (torch.tensor): Batch of corevectors, will be parsed with self.parser (see __init__()).
        - verbose (bool): Print progress messages.
        
        '''
        cvs = kwargs['cvs']
        verbose = kwargs.get('verbose', False) 

        # # check for empiracal posterios `_empp`
        if self._empp == None:
            raise RuntimeError('No prediction probabilities. Please run classifiers[layer].compute_empirical_posteriors() first.')
        data = self.parser(cvs=cvs)
        cp = self.classifier_probabilities(data=data, verbose=verbose).to(self.device)
        lp = cp@self._empp
        lp /= lp.sum(dim=1, keepdim=True)

        return lp
