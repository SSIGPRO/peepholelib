from time import time

# our stuff
from .cdc_base import CDCBase

# torch stuff
import torch

# https://github.com/CSOgroup/torchgmm/tree/main
from torchgmm.bayes import GaussianMixture as tGMM

import logging
logging.getLogger('pytorch_lightning.utilities.rank_zero').setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.CRITICAL)

class ClassDependentGMM(CDCBase):
    def __init__(self, **kwargs):
        cls_kwargs = kwargs.pop('cls_kwargs') if 'cls_kwargs' in kwargs else {}
        CDCBase.__init__(self, **kwargs)

        # one GMM instance per model class
        self._classifiers = [
            tGMM(
                num_components=self.nl_class,
                **cls_kwargs,
                trainer_params=dict(
                    num_nodes=1,
                    max_epochs=50,
                    accelerator=self.device.type,
                    devices=[self.device.index],
                    enable_progress_bar=False
                )
            )
            for _ in range(self.nl_model)
        ]

        self._clas_path = self.path / self.name
        return

    def fit(self, **kwargs):
        '''
        Fit one GMM per class, using only data from that class.

        Args:
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): Parsed datasets respective the `coreVectors`.
        - corevectors (peepholelib.coreVectors.coreVectors.CoreVectors): Corevectors respective the `datasets`.
        - loader (str): Which loader used for fitting. Defaults to 'train'.
        - verbose (bool): Print progress messages.
        '''
        _dss = kwargs['datasets']
        _cvs = kwargs['corevectors']
        loader = kwargs.get('loader', 'train')
        verbose = kwargs.get('verbose', False)

        cvs = _cvs._corevds[loader][self.target_module]
        data = self.parser(cvs=cvs)
        labels = _dss._dss[loader][self.label_key]

        if verbose: print('\n ---- Class-Dependent-Classifier GMM (one GMM per class)\n')

        for c in range(self.nl_model):
            class_data = data[labels == c]

            if class_data.shape[1] != self.n_features:
                raise RuntimeError(
                    f'Data has shape {data.shape} after parsing, '
                    f'but n_features={self.n_features} was expected.'
                )

            if verbose: print(f'Fitting GMM for class {c} ({class_data.shape[0]} samples)')

            converged = False
            t0 = time()
            while not converged:
                fit_data = class_data.clone().detach()
                self._classifiers[c].fit(fit_data)
                t1 = time()
                converged = not self._classifiers[c].predict_proba(fit_data[0:1]).isnan().any()
                if verbose:
                    if not converged: print(f'GMM for class {c} failed to converge, retrying. fitting time = {t1-t0}')
                    else: print(f'fitting time = {t1-t0}')
                t0 = t1
            # self.save_single_class(c=c) # testing
        self.save()
        return

    def classifier_probabilities(self, **kwargs):
        '''
        Compute the probability of each sample under each class GMM.
        Uses per-class log-densities (score_samples), normalized via softmax.

        Args:
        - data (torch.Tensor): Parsed corevectors of shape (N, n_features).

        Returns:
        - probs (torch.Tensor): Shape (N, nl_model), normalized class probabilities.
        '''
        data = kwargs['data']
        
        log_densities = torch.stack(
            [self._classifiers[c].score_samples(data).flatten() for c in range(self.nl_model)],
            dim=1
        ) # flatten is new
        
        return log_densities

    def predict(self, data):
        probs = self.classifier_probabilities(data=data)
        return probs.argmax(dim=1)
       
    def save(self, **kwargs):
        self._clas_path.mkdir(parents=True, exist_ok=True)
        for c, _ in enumerate(self._classifiers):
            self._classifiers[c].save(self._clas_path / f'class_{c}')
        return

    def load(self, **kwargs):
        if self._clas_path.exists():
            self._classifiers = [
                tGMM.load(self._clas_path / f'class_{c}')
                for c in range(self.nl_model)
            ]
            ok = True
        else:
            ok = False
        return ok
