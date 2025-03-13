# our stuff
from .classifier_base import ClassifierBase

# torch stuff
import torch
from torch.utils.data import DataLoader

# torch kmeans

# https://github.com/CSOgroup/torchgmm/tree/main
from torchgmm.clustering import KMeans as tKMeans

import logging
logging.getLogger('pytorch_lightning.utilities.rank_zero').setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.CRITICAL)

class KMeans(ClassifierBase): # quella buona
    def __init__(self, **kwargs):
        cls_kwargs = kwargs.pop('cls_kwargs') if 'cls_kwargs' in kwargs else {}
        ClassifierBase.__init__(self, **kwargs)

        self._classifier = tKMeans(num_clusters=self.nl_class, **cls_kwargs, trainer_params=dict(num_nodes=1, accelerator=self.device.type, devices=[self.device.index], max_epochs=5000, enable_progress_bar=True))

    def fit(self, **kwargs):
        '''
        Fitss clusters.
        Args:
        '''
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        
        if self._cvs == None:
            raise RuntimeError('No corevectors found. Instantiate the class or run `set_corevectors()` passing the `corevectors` argument.')

        if verbose: 
            print('\n ---- KMeans classifier\n')
            print('Parsing data')

        # temp dataloader for loading the whole dataset
        data = self.parser(cvs=self._cvs._cvsds, **self.parser_kwargs)
        
        if data.shape[1] != self.n_features:
            raise RuntimeError('Something is weird...\n Data has shape {data.shape} after parsing corevectors with the parser {self.parser}\nWhile n_features={self.n_features} was passed during construction.')


        if verbose: print('Fitting KMeans')
        self._classifier.fit(data)

        return
    
    def classifier_probabilities(self, **kwargs):
        '''
        Get prediction probabilities based on the fitted modelfor the provided inputs.
        
        Args:
        - batch: data batch containing data to be parsed with the paser function set on __init__() 
        '''
        
        cvs = kwargs['cvs']

        data = self.parser(cvs=cvs, **self.parser_kwargs)
        distances = torch.tensor(self._classifier.transform(data), dtype=data.dtype)
        
        # changing strategy: back to softmin
        probs = torch.nn.functional.softmin(distances, dim=1)
            
        return probs 
    
    def save(self, **kwargs):
        self.path.mkdir(parents=True, exist_ok=True)
        
        if not self.file_path == None:
            self._clas_path = self.path/('{self.name}.' + f'n_features={self.n_features}' + '.' + f'nl_class={self.nl_class}'+'.'+'{nl_model={self.nl_model}')

        self.file_path.mkdir(parents=True, exist_ok=True)
        self._classifier.save(self._clas_path)
        super().save()
    
        return

    def load(self, **kwargs):
        if not self.file_path == None:
            self._clas_path = self.path/('{self.name}.' + f'n_features={self.n_features}' + '.' + f'nl_class={self.nl_class}'+'.'+'{nl_model={self.nl_model}')

        self._classifier.load(self._clas_path)
        super().load()

        return