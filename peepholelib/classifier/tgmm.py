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
        ClassifierBase.__init__(self, **kwargs)
        
        '''
        load = kwargs['load']# if 'load' in kwargs else False
        if not load:
        ''' 
        self._classifier = tGMM(num_components=self.nl_class, **cls_kwargs, trainer_params=dict(num_nodes=1, accelerator=self.device.type, devices=[self.device.index], max_epochs=5000, enable_progress_bar=True))
        return

    def fit(self, **kwargs):
        '''
        Fit GMM. 
        
        Args:
        '''
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        _cvs_dl = kwargs['cvs']
        _act_dl = kwargs['act']
        
        if verbose: 
            print('\n ---- GMM classifier\n')
            print('Parsing data')

        # temp dataloader for loading the whole dataset
        data = self.parser(cvs=_cvs_dl.dataset, **self.parser_kwargs)

        if verbose: print('Fitting GMM')
        
        self._classifier.fit(data)
        
        self._cvs_dl = _cvs_dl
        self._act_dl = _act_dl
        return
    
    def classifier_probabilities(self, **kwargs):
        '''
        Get prediction probabilities based on the fitted modelfor the provided inputs.
        
        Args:
        - cvs: data containing data to be parsed with the paser function set on __init__() 
        '''
        
        cvs = kwargs['cvs']

        data = self.parser(cvs=cvs, **self.parser_kwargs)
        probs = torch.tensor(self._classifier.predict_proba(data), dtype=data.dtype)
        return probs  
            
