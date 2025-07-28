# our stuff
from .classifier_base import ClassifierBase

# torch stuff
import torch
from torch.nn.functional import softmax as sm

# BIRCH 
import numpy as np
from sklearn.cluster import BisectingKMeans as skBirch

class Birch(ClassifierBase): # quella buona
    def __init__(self, **kwargs):
        cls_kwargs = kwargs.pop('cls_kwargs') if 'cls_kwargs' in kwargs else {}
        ClassifierBase.__init__(self, **kwargs)

        self._classifier = skBirch(
                n_clusters = self.nl_class,
                **cls_kwargs,
                )

        self._clas_path = self.path/(self.name+'.Birch'+self._suffix)
        self._empp_file = self._clas_path/'empp.pt'

        return

    def predict(self, data):
        return self._classifier.predict(data.detach().cpu().double())

    def fit(self, **kwargs):
        '''
        Fit GMM. 

        Args:
        - corevectors (TensorDict): Corevectors.
        - loader (str): Which loader used for fitting the GMM, usually 'train'. Defaults to 'train'. 
        - verbose (Bool): Print progress messages. 
        '''
        _cvs = kwargs.get('corevectors')
        loader = kwargs.get('loader', 'train')
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        
        cvs = _cvs._corevds[loader]

        if verbose: print('\n ---- Birch classifier\n')

        # temp dataloader for loading the whole dataset
        data = self.parser(cvs=cvs)
        
        if data.shape[1] != self.n_features:
            raise RuntimeError(f'Something is weird...\n Data has shape {data.shape} after parsing corevectors with the parser {self.parser}\nWhile n_features={self.n_features} was passed during construction.')

        if verbose: print('Fitting Birch')
        #self._classifier.fit(data)
        self._classifier.fit(data.detach().cpu().double())
        
        return
    
    def classifier_probabilities(self, **kwargs):
        '''
        Get prediction probabilities based on the fitted model for the provided inputs.
        
        Args:
        - data (TensorDict): data containing data to be parsed with the paser function set on __init__() 
        '''
        
        data = kwargs['data']

        dists = self._classifier.transform(data.detach().cpu().double())
        dists /= torch.tensor(dists).sum(dim=-1, keepdims=True)
        probs = 1 - dists 

        return probs.float()   
    
    def save(self, **kwargs):
        self._clas_path.mkdir(parents=True, exist_ok=True)
        np.save(self._clas_path.as_posix()+'/params.npy', self._classifier.get_params())
        
        super().save()
        
        return

    def load(self, **kwargs):
        params = np.load(self._clas_path.as_posix()+'/params.npy', allow_pickle=True)
        print(params)
        self._classifier.set_params(params)
        super().load()
        
        return
