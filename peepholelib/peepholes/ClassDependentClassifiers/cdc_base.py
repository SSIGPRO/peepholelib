# python stuff
import abc  
from pathlib import Path
from tqdm import tqdm

# torch stuff
import torch
from torch.utils.data import DataLoader
from peepholelib.peepholes.drill_base import DrillBase

class CDCBase(DrillBase, metaclass=abc.ABCMeta): 
    def __init__(self, **kwargs):
        DrillBase.__init__(self, **kwargs)

        # number of classes in classifier a.k.a. number of clusters
        self.nl_class = kwargs['nl_classifier']
        self.label_key = kwargs.get('label_key', 'label')
        self.reducer = kwargs['reducer']

        self.parser = self.reducer.parser 
        # computed in inheriting classes 
        self._classifiers = None
        self._classifiers_test = None

        # defined in __init__(), used in save() and load()
        self._clas_path = None

        return
    
    @abc.abstractmethod
    def load(self, **kwargs):
        return 

    @abc.abstractmethod
    def save(self, **kwargs):
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
    
    def __call__(self, **kwargs):
        '''
        Compute class probabilities directly from per-class GMM densities.

        Args:
        - cvs (torch.tensor): Batch of corevectors, parsed with self.parser.
        - verbose (bool): Print progress messages.
        '''
        cvs = kwargs['cvs']

        data = self.parser(cvs=cvs).to(self.device)
        lp = self.classifier_probabilities(data=data).to(self.device)
        return lp
