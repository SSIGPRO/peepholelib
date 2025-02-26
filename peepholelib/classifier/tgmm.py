# our stuff
from .classifier_base import ClassifierBase

# torch stuff
import torch
from torch.utils.data import DataLoader

# https://github.com/CSOgroup/torchgmm/tree/main
from torchgmm.bayes import GaussianMixture as tGMM

import logging
logging.getLogger('pytorch_lightning.utilities.rank_zero').setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.CRITICAL)

class GMM(ClassifierBase): # quella buona
    def __init__(self, **kwargs):
        cls_kwargs = kwargs.pop('cls_kwargs') if 'cls_kwargs' in kwargs else {}
        
        load = kwargs['load']# if 'load' in kwargs else False
        
        ClassifierBase.__init__(self, **kwargs)

        if not load:
            self._classifier = tGMM(num_components=self.nl_class, **cls_kwargs, trainer_params=dict(num_nodes=1, accelerator=self.device.type, devices=[self.device.index], max_epochs=5000, enable_progress_bar=True))
        

    def fit(self, **kwargs):
        '''
        Fit GMM. 
        
        Args:
        '''
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        _dl = kwargs['dataloader']
        
        if verbose: 
            print('\n ---- GMM classifier\n')
            print('Parsing data')

        # temp dataloader for loading the whole dataset
        data, _ = self.parser(data=_dl.dataset, **self.parser_kwargs)

        if verbose: print('Fitting GMM')
        
        self._classifier.fit(data)
        
        self._fit_dl = _dl
        return
    
    def classifier_probabilities(self, **kwargs):
        '''
        Get prediction probabilities based on the fitted modelfor the provided inputs.
        
        Args:
        - batch: data containing data to be parsed with the paser function set on __init__() 
        '''
        
        batch = kwargs['batch']

        data, _ = self.parser(data = batch, **self.parser_kwargs)
        probs = torch.tensor(self._classifier.predict_proba(data), dtype=data.dtype)
        return probs  
            
