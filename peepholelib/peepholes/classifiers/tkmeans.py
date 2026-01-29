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

        self._clas_path = self.path/self.name
        self._empp_file = self._clas_path/f'empp_{self.label_key}.pt'
        return

    def fit(self, **kwargs):
        '''
        Fitss clusters.
        Args:
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): Parsed datasets respective the `coreVectors`.
        - corevectors (peepholelib.coreVectors.coreVectors.CoreVectors): Corevectors respective the `datasets`.
        - loader (str): Which loader used for fitting the GMM, usually 'train'. Defaults to 'train'. 
        - batch_size: Do the computation in batchs. Defaults to 512.
<<<<<<< HEAD
        - compute_empp (bool): Wether to compute the empirical posterior. Defaults to `True`.
=======
>>>>>>> 0eef6bb (implement svg kernel svd (#127))
        - verbose (Bool): Print progress messages. 
        '''
        _dss = kwargs['datasets']
        _cvs = kwargs['corevectors']
        loader = kwargs.get('loader', 'train')
        bs = kwargs.get('batch_size', 512)
<<<<<<< HEAD
        _compute_empp = kwargs.get('compute_empp', True)
=======
>>>>>>> 0eef6bb (implement svg kernel svd (#127))
        verbose = kwargs.get('verbose', False)

        cvs = _cvs._corevds[loader][self.target_module]
        if verbose: 
            print('\n ---- KMeans classifier\n')

        # temp dataloader for loading the whole dataset
        data = self.parser(cvs=cvs)
        
        if data.shape[1] != self.n_features:
            raise RuntimeError(f'Something is weird...\n Data has shape {data.shape} after parsing corevectors with the parser {self.parser}\nWhile n_features={self.n_features} was passed during construction.')


        if verbose: print('Fitting KMeans')
        self._classifier.fit(data)

        # compute empirical posteriors       
<<<<<<< HEAD
        if _compute_empp:
            self._compute_empirical_posteriors(
                    datasets = _dss,
                    corevectors = _cvs,
                    loader = loader,
                    batch_size = bs,
                    verbose = verbose
                    )
=======
        self._compute_empirical_posteriors(
                datasets = _dss,
                corevectors = _cvs,
                loader = loader,
                bs = bs,
                verbose = verbose
                )
>>>>>>> 0eef6bb (implement svg kernel svd (#127))
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
    
    def predict(self, data):
        return self._classifier.predict(data)

    def save(self, **kwargs):
        self._clas_path.mkdir(parents=True, exist_ok=True)
        self._classifier.save(self._clas_path)
        super().save()
    
        return

    def load(self, **kwargs):
        if self._clas_path.exists(): 
<<<<<<< HEAD
            self._classifier = tKMeans.load(self._clas_path)
            super().load()
            ok = True 
=======
            self._classifier = tGMM.load(self._clas_path)
            ok = super().load()
>>>>>>> 0eef6bb (implement svg kernel svd (#127))
        else:
            ok = False

        return ok
