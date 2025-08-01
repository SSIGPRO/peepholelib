# our stuff
from .classifier_base import ClassifierBase

# torch stuff
import torch

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

        self._classifier = tKMeans(
                num_clusters = self.nl_class,
                **cls_kwargs,
                trainer_params = dict(
                    num_nodes = 1,
                    accelerator = self.device.type,
                    devices = [self.device.index],
                    max_epochs = 50000,
                    enable_progress_bar = False
                    )
                )

        self._clas_path = self.path/(self.name+'.KMeans'+self._suffix)
        self._empp_file = self._clas_path/'empp.pt'
        return

    def fit(self, **kwargs):
        '''
        Fitss clusters.
        Args:
        - corevectors (TensorDict): Corevectors.
        - loader (str): Which loader used for fitting the GMM, usually 'train'. Defaults to 'train'. 
        - verbose (Bool): Print progress messages. 
        '''
        _cvs = kwargs.get('corevectors')
        loader = kwargs.get('loader', 'train')
        verbose = kwargs.get('verbose', False)

        cvs = _cvs._corevds[loader]
        if verbose: 
            print('\n ---- KMeans classifier\n')

        # temp dataloader for loading the whole dataset
        data = self.parser(cvs=cvs)
        
        if data.shape[1] != self.n_features:
            raise RuntimeError(f'Something is weird...\n Data has shape {data.shape} after parsing corevectors with the parser {self.parser}\nWhile n_features={self.n_features} was passed during construction.')


        if verbose: print('Fitting KMeans')
        self._classifier.fit(data)

        return
    
    def classifier_probabilities(self, **kwargs):
        '''
        Get prediction probabilities based on the fitted modelfor the provided inputs.
        
        Args:
        - data (TensorDict): data containing data to be parsed with the paser function set on __init__() 
        '''
        
        data = kwargs['data']

        distances = self._classifier.transform(data)
        
        # changing strategy: back to softmin
        probs = torch.nn.functional.softmin(distances, dim=1)
            
        return probs 
    
    def save(self, **kwargs):
        self._clas_path.mkdir(parents=True, exist_ok=True)
        self._classifier.save(self._clas_path)
        super().save()
    
        return

    def load(self, **kwargs):
        self._classifier = tKMeans.load(self._clas_path)
        super().load()

        return
