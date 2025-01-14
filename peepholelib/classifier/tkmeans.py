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
        _dl = kwargs['dataloader']

        if verbose: 
            print('\n ---- KMeans classifier\n')
            print('Parsing data')

        print('tkmeans device: ', self.device)

        # temp dataloader for loading the whole dataset
        data = self.parser(data=_dl.dataset, **self.parser_kwargs)
        print('data shape: ', data.shape, type(data))

        if verbose: print('Fitting KMeans')
        self._classifier.fit(data)
        
        self._fit_dl = _dl
        return
    
    def classifier_probabilities(self, **kwargs):
        '''
        Get prediction probabilities based on the fitted modelfor the provided inputs.
        
        Args:
        - batch: data batch containing data to be parsed with the paser function set on __init__() 
        '''
        
        batch = kwargs['batch']

        data = self.parser(data = batch, **self.parser_kwargs)
        distances = torch.tensor(self._classifier.transform(data), dtype=data.dtype)

        # convert distances to probabilities (soft assignment) Gaussian-like softmax
        # TODO: Check the var in the exponent. Should it be over the training set? Should it be there?
        #probs = torch.exp(-distances ** 2 / (2 * (distances.std() ** 2)))
        #probs = torch.exp(-distances ** 2 / 2 )

        # normalize to probabilities
        #probs /= probs.sum(dim=1, keepdim=True)

        # changing strategy: back to softmin
        probs = torch.nn.functional.softmin(distances, dim=1)
            
        return probs 
