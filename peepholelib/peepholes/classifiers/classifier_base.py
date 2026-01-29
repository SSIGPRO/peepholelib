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

<<<<<<< HEAD
        self.parser = self.reducer.parser 
=======
        self.parser = self.reducer.parser

>>>>>>> 0eef6bb (implement svg kernel svd (#127))
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
<<<<<<< HEAD
        return 

    @abc.abstractmethod
    def save(self, **kwargs):
        if self._empp != None:
            torch.save(self._empp, self._empp_file)
=======
            ok = True
        else:
            ok = False
        return ok 

    @abc.abstractmethod
    def save(self, **kwargs):
        torch.save(self._empp, self._empp_file)
>>>>>>> 0eef6bb (implement svg kernel svd (#127))
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
<<<<<<< HEAD
        - loader (str): Which loader used for computing the Empirical Posteriors, usually 'train'. Defaults to 'train'.
=======
        - loader (str): Which loader used for computing the Empirical Posteriors, usually 'train'. Defaults to 'train'. 
        - batch_size: Do the computation in batchs. Defaults to 512.
>>>>>>> 0eef6bb (implement svg kernel svd (#127))
        - verbose (Bool): Print progress messages. 
        '''
        
        dss = kwargs['datasets']
        cvs = kwargs['corevectors']
        loader = kwargs.get('loader', 'train')
<<<<<<< HEAD
=======
        bs = kwargs.get('batch_size', 512)
        verbose = kwargs.get('verbose', False)
>>>>>>> 0eef6bb (implement svg kernel svd (#127))

        # pre-allocate empirical posteriors and cluster population counts
        _empp = torch.zeros(self.nl_class, self.nl_model)
        _counts = torch.zeros(self.nl_class, 1)
        
        # create dataloaders
        dss_dl = DataLoader(dss._dss[loader], batch_size=bs, collate_fn=lambda x: x, shuffle=False)
        cvs_dl = DataLoader(cvs._corevds[loader], batch_size=bs, collate_fn=lambda x: x, shuffle=False)

        # iterate over _fit_data
        if verbose: print('Computing empirical posterior')
        for _dss, _cvs in tqdm(zip(dss_dl, cvs_dl), disable=not verbose):
            data, label = self.parser(cvs=_cvs[self.target_module], dss=_dss, label_key = self.label_key)
            data, label = data.to(self.device), label.to(self.device)

            preds = self.predict(data)
            for p, l in zip(preds, label):
                _empp[int(p), int(l)] += 1
                _counts[int(p)] += 1

        # normalize each row by the number of samples assigned to that cluster
        _empp = _empp / _counts.clamp_min(1.0)

        # replace NaN with 0
        self._empp = torch.nan_to_num(_empp).to(self.device)
        
        return 
    
    def _compute_concept_empirical_posteriors(self, **kwargs):
        '''
        Compute the empirical posterior matrix P for concepts/attributes, where
        P(g, k) is the probability that a sample assigned to classifier's cluster g
        has attribute k = 1.
        P(g, k) = P(attribute_k=1 | cluster=g)

        Args:
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): Parsed datasets respective the `coreVectors`.
        - corevectors (peepholelib.coreVectors.coreVectors.CoreVectors): Corevectors respective the `datasets`.
        - loader (str): Which loader used for computing the Empirical Posteriors, usually 'train'. Defaults to 'train'.
        - batch_size: Do the computation in batchs. Defaults to 64.
        - verbose (Bool): Print progress messages.
        '''

        dss = kwargs['datasets']
        cvs = kwargs['corevectors']
        loader = kwargs.get('loader', 'train')
        bs = kwargs.get('batch_size', 64)
        verbose = kwargs.get('verbose', False)

        # pre-allocate empirical posteriors: [n_clusters, n_concepts]
        _empp = torch.zeros(self.nl_class, self.nl_model, device=self.device)
        _counts = torch.zeros(self.nl_class, 1, device=self.device)

        # create dataloaders
        dss_dl = DataLoader(dss._dss[loader], batch_size=bs, collate_fn=lambda x: x, shuffle=False)
        cvs_dl = DataLoader(cvs._corevds[loader], batch_size=bs, collate_fn=lambda x: x, shuffle=False)

        # attribute names 

        # iterate over data
        if verbose: print('Computing empirical posterior (concepts)')
        concept_keys = None
        for _dss, _cvs in tqdm(zip(dss_dl, cvs_dl), disable=not verbose):
            if concept_keys is None:
                sample0 = _dss[0]
                excluded = {'image', 'label', 'output', 'result', 'pred', 'bbox'}
                concept_keys = sorted([
                    k
                    for k, v in sample0.items()
                    if (
                        k not in excluded
                        and torch.is_tensor(v)
                        and v.ndim == 0
                    )
                ])

                if len(concept_keys) != self.nl_model:
                    if verbose: print(f'Inferred {len(concept_keys)} concept keys, but nl_model={self.nl_model}')
            # parse corevectors
            data, _ = self.parser(cvs=_cvs[self.target_module], dss=_dss, label_key = self.label_key)
            data = data.to(self.device)

            # build [B, K] attribute matrix from batch dicts
            A = torch.stack([
                torch.stack([sample[k] for k in concept_keys]).float()
                for sample in _dss
            ], dim=0).to(self.device)  # [B, K]

            # hard cluster assignment
            preds = self.predict(data)  # [B]

            # accumulate attribute sums per cluster, and counts per cluster
            for p, a in zip(preds, A):
                _empp[int(p)] += a
                _counts[int(p)] += 1

        # normalize by number of samples per cluster 
        _empp = _empp / _counts.clamp_min(1.0)

        #_empp = torch.nan_to_num(_empp)
        self._empp = _empp
        return

    def _compute_empirical_posteriors2(self, **kwargs):
        '''
        Compute the empirical posterior-like matrix for class labels, where rows are
        classifier clusters and columns are model classes.

        This follows the same batching and accumulation logistics of
        `_compute_concept_empirical_posteriors`, but uses labels instead of concepts.

        Args:
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): Parsed datasets respective the `coreVectors`.
        - corevectors (peepholelib.coreVectors.coreVectors.CoreVectors): Corevectors respective the `datasets`.
        - loader (str): Which loader used for computing the Empirical Posteriors, usually 'train'. Defaults to 'train'.
        - batch_size: Do the computation in batchs. Defaults to 64.
        - label_key (str): Label key from the parsed dataset. Defaults to `self.label_key`.
        - n_label_classes (int): Number of output columns (classes) for one-hot labels. Defaults to `self.nl_model`.
        - verbose (Bool): Print progress messages.
        '''
        dss = kwargs['datasets']
        cvs = kwargs['corevectors']
        loader = kwargs.get('loader', 'train')
        bs = kwargs.get('batch_size', 64)
        label_key = kwargs.get('label_key', self.label_key)
        n_label_classes = kwargs.get('n_label_classes', self.nl_model)
        verbose = kwargs.get('verbose', False)

        # pre-allocate empirical posteriors: [n_clusters, n_classes]
        _empp = torch.zeros(self.nl_class, n_label_classes, device=self.device)
        _counts = torch.zeros(self.nl_class, 1, device=self.device)

        # create dataloaders
        dss_dl = DataLoader(dss._dss[loader], batch_size=bs, collate_fn=lambda x: x, shuffle=False)
        cvs_dl = DataLoader(cvs._corevds[loader], batch_size=bs, collate_fn=lambda x: x, shuffle=False)

        # iterate over data
        if verbose: print('Computing empirical posterior (classes)')
        for _dss, _cvs in tqdm(zip(dss_dl, cvs_dl), disable=not verbose):
            # parse corevectors
            data, _ = self.parser(cvs=_cvs[self.target_module], dss=_dss, label_key=label_key)
            data = data.to(self.device)

            labels = _dss[label_key].to(self.device)
            if labels.ndim == 1:
                labels = torch.nn.functional.one_hot(labels.long(), num_classes=n_label_classes).float()
            else:
                labels = labels.float()

            # hard cluster assignment
            preds = self.predict(data)  # [B]

            # accumulate class sums per cluster, and counts per cluster
            for p, l in zip(preds, labels):
                _empp[int(p)] += l
                _counts[int(p)] += 1

        # normalize by number of samples per cluster
        _empp = _empp / _counts.clamp_min(1.0)

        #_empp = torch.nan_to_num(_empp)
        self._empp = _empp
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
        if self._empp is None:
            raise RuntimeError('No prediction probabilities. Please run classifiers[layer].compute_empirical_posteriors() first.')
        data = self.parser(cvs=cvs)
        cp = self.classifier_probabilities(data=data, verbose=verbose).to(self.device)
        lp = cp@self._empp
        #lp /= lp.sum(dim=1, keepdim=True)

        return lp
