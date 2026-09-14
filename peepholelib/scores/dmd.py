from math import floor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

import torch
from peepholelib.scores.score import Score

class DMDBase(Score):
    '''
    Compute the DMD score from the corevectors of a layer. `fit()` must be called once before scoring.

    The corevectors are taken as they were saved, so the score is agnostic to the dimensionality reduction used to compute them, the `Null` reducer included, and applies to any target module.

    With `normalize=True` the corevectors are L2-normalized before the statistics are computed and before scoring, which gives the Mahalanobis++ score.

    References:
    - DMD: Lee et al., https://arxiv.org/abs/1807.03888
    - Mahalanobis++: https://arxiv.org/abs/2505.18032

    Args:
    - normalize (bool): L2-normalize the corevectors, giving the Mahalanobis++ score. Defaults to `False`.
    - corevectors (peepholelib.coreVectors.coreVectors.CoreVectors): corevectors.
    - layer (str): target module whose corevectors are scored. Should be the same one passed to `fit()`.
    - loaders (list[str]): loaders to consider. If `None`, gets all loaders in `corevectors._corevds`. Defaults to `None`.
    - device (torch.device): device to perform the computations.
    - verbose (bool): print progress messages.
    '''

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'DMD-B')
        Score.__init__(self, **{k: v for k, v in kwargs.items() if k != 'normalize'})

        # the corevectors are L2-normalized in both fit() and compute(), so it belongs to the score
        self.normalize = kwargs.get('normalize', False)

        # computed in fit()
        self._means = None
        self._precision = None
        return

    def fit(self, **kwargs):
        '''
        Compute the class-conditional means and the shared precision matrix from the corevectors of `fit_key`, L2-normalized when `self.normalize`.

        Args:
        - corevectors (peepholelib.coreVectors.coreVectors.CoreVectors): corevectors.
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed datasets, providing the labels.
        - layer (str): target module whose corevectors are used as features.
        - n_classes (int): number of classes.
        - fit_key (str): loader key used for fitting. Defaults to `'train'`.
        - label_key (str): key used to read the class label from the dataset. Defaults to `'label'`.
        - device (torch.device): device to perform the computations.
        - verbose (bool): print progress messages.
        '''
        cvs = kwargs['corevectors']
        dss = kwargs['datasets']
        layer = kwargs['layer']
        n_classes = kwargs['n_classes']
        fit_key = kwargs.get('fit_key', 'train')
        label_key = kwargs.get('label_key', 'label')
        device = kwargs.get('device')
        verbose = kwargs.get('verbose', False)

        if verbose: print('Fitting', self.name, 'on dataset', fit_key)

        _cvs = cvs._corevds[fit_key][layer].to(device).float()

        if self.normalize:
            _cvs = _cvs/_cvs.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)

        labels = dss._dss[fit_key][:][label_key].long().to(_cvs.device)
        n_samples, n_features = _cvs.shape

        counts = torch.zeros(n_classes, dtype=torch.long, device=_cvs.device)
        sums = torch.zeros(n_classes, n_features, dtype=torch.float64, device=_cvs.device)

        counts.index_add_(0, labels, torch.ones_like(labels))
        sums.index_add_(0, labels, _cvs.double())

        means = torch.zeros(n_classes, n_features, device=_cvs.device)
        non_empty = counts > 0
        means[non_empty] = (sums[non_empty]/counts[non_empty].unsqueeze(1)).float()

        centered = (_cvs - means[labels]).double()
        covariance_matrix = centered.t()@centered/n_samples

        try:
            precision = torch.linalg.pinv(covariance_matrix, hermitian=True).float()
        except TypeError:
            precision = torch.linalg.pinv(covariance_matrix).float()

        self._means = means.cpu()
        self._precision = precision.cpu()
        self._save_fitting()
        return

    def compute(self, **kwargs):
        if self._means is None:
            raise RuntimeError(f'{self.name} statistics not computed. Please run fit() first.')

        cvs = kwargs['corevectors']
        layer = kwargs['layer']
        loaders = kwargs.get('loaders') or list(cvs._corevds.keys())
        device = kwargs.get('device')
        verbose = kwargs.get('verbose', False)

        # skip the loaders already computed
        loaders = [k for k in loaders if not self._is_computed(ds_key=k)]

        means = self._means.to(device)
        precision = self._precision.to(device)
        n_classes = means.shape[0]

        for ds_key in loaders:
            if verbose: print('Computing', self.name, 'for dataset', ds_key)

            _cvs = cvs._corevds[ds_key][layer].to(device).float()

            if self.normalize:
                _cvs = _cvs/_cvs.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)

            gaussian_score = torch.zeros(len(_cvs), n_classes, device=_cvs.device)
            for c in range(n_classes):
                zero_f = _cvs - means[c]

                # only the diagonal of zero_f@precision@zero_f.t() is needed
                gaussian_score[:, c] = -0.5*((zero_f@precision)*zero_f).sum(dim=1)

            scores = gaussian_score.max(dim=1)[0].detach().cpu().reshape(-1)
            self._record(ds_key=ds_key, scores=scores)

        return self._df

    def _save_fitting(self, **kwargs):
        self.path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'means': self._means,
            'precision': self._precision,
            'normalize': self.normalize,
            }, self._fit_file)
        return

    def load(self, **kwargs):
        if not self._fit_file.exists():
            return 0

        fitting = torch.load(self._fit_file, weights_only=False)

        # the statistics are only valid for the features they were fitted on
        if fitting['normalize'] != self.normalize:
            raise RuntimeError(f'{self.name} was fitted with normalize={fitting["normalize"]}, but normalize={self.normalize} was passed. Please restart with normalize={fitting["normalize"]}.')

        self._means = fitting['means']
        self._precision = fitting['precision']
        return 1

class DMDScore(Score):
    '''
    Compute the DMD score with one linear regressor per negative loader, trained on the positive samples of `pos_train_loader` against a balanced draw of the negative ones.

    Since the scores of the positive samples change for each negative loader used in training, the negative test loader they were trained against is recorded in the `'calib key'` column.

    Args:
    - peepholes (peepholelib.peepholes.peepholes.Peepholes): peepholes from which the features are extracted.
    - pos_test_loader (str): loader to consider as positive samples for testing. Defaults to `'test'`.
    - target_modules (list[str]): list of target modules, as keys from the model `state_dict`. Should be the same ones passed to `fit()`.
    - verbose (bool): print progress messages.
    '''

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'DMD')
        Score.__init__(self, **kwargs)

        # computed in fit(), keyed by the negative TEST loader they will score
        self._lrs = {}
        self._scalers = {}

        # set in fit()
        self._fitted = False
        return

    def fit(self, **kwargs):
        '''
        Train one logistic regressor for each negative loader against `pos_train_loader`.
        The regressors are kept in `self._lrs`, keyed by the negative TEST loader they will be used to score.

        Pairs whose scores are already in `self.df` are skipped.

        Args:
        - peepholes (peepholelib.peepholes.peepholes.Peepholes): peepholes from which we compute the linear regressors.
        - pos_train_loader (str): loader to consider as positive samples for training. Typically in-distribution for OOD or original samples for attacks. Defaults to `'val'`.
        - neg_loaders (dict{str: list[str]}): dictionary with keys for negative samples. The key corresponds to the TEST loader, and the value is a list of loaders used as negative samples for training.
        - target_modules (list[str]): list of target modules, as keys from the model `state_dict`. If `None`, uses all modules in `peepholes._phs[pos_train_loader]`. Defaults to `None`.
        - scaling (bool): standardize the features before fitting the regressor. Defaults to `False`.
        - verbose (bool): print progress messages.
        '''
        phs = kwargs['peepholes']
        pos_train_key = kwargs.get('pos_train_loader', 'val')
        neg_keys = kwargs['neg_loaders']
        target_modules = kwargs.get('target_modules') or list(phs._phs[pos_train_key].keys())
        scaling = kwargs.get('scaling', False)
        verbose = kwargs.get('verbose', False)

        pending_neg = {
                k: v for k, v in neg_keys.items()
                if not self._is_computed(ds_key=k, calib_key=k)
                }
        if len(pending_neg) == 0:
            self._fitted = True
            return
        print(f'{self.name}: {len(pending_neg)}/{len(neg_keys)} negative loaders still to fit: {list(pending_neg.keys())}')

        train_pos = torch.stack([phs._phs[pos_train_key][layer].max(dim=1)[0] for layer in target_modules], dim=1)
        n_pos = len(train_pos)

        for neg_test_key, neg_train_keys in pending_neg.items():
            if verbose: print('Fitting', self.name, 'for dataset', neg_test_key)

            n_neg_keys = len(neg_train_keys)
            n_per_loader = floor(n_pos/n_neg_keys)

            # get n_per_loader samples for each negative loader
            train_neg = []
            for nl in neg_train_keys:
                _train_neg = torch.stack([phs._phs[nl][layer].max(dim=1)[0] for layer in target_modules], dim=1)
                idx = torch.randperm(len(_train_neg))
                train_neg.append(_train_neg[idx[:n_per_loader]])
            train_neg = torch.vstack(train_neg)

            # train data and labels
            train_data = torch.vstack((train_pos, train_neg))
            train_label = torch.hstack((torch.ones(len(train_pos)), torch.zeros(len(train_neg))))

            if scaling:
                self._scalers[neg_test_key] = StandardScaler()
                train_data = self._scalers[neg_test_key].fit_transform(train_data)

            # You can use torch tensors for the LogisticRegressionCV, no need to numpy
            self._lrs[neg_test_key] = LogisticRegressionCV(n_jobs=-1, max_iter=5000).fit(train_data, train_label)

        self._fitted = True
        self._save_fitting()
        return

    def compute(self, **kwargs):
        if not self._fitted:
            raise RuntimeError(f'{self.name} regressors not computed. Please run fit() first.')

        phs = kwargs['peepholes']
        pos_test_key = kwargs.get('pos_test_loader', 'test')
        target_modules = kwargs.get('target_modules') or list(phs._phs[pos_test_key].keys())
        verbose = kwargs.get('verbose', False)

        test_pos = torch.stack([phs._phs[pos_test_key][layer].max(dim=1)[0] for layer in target_modules], dim=1)

        for neg_test_key, lr in self._lrs.items():
            if self._is_computed(ds_key=neg_test_key, calib_key=neg_test_key):
                if verbose: print(self.name, neg_test_key, 'already computed, skipping')
                continue

            if verbose: print('Computing', self.name, 'for dataset', neg_test_key)

            test_neg = torch.stack([phs._phs[neg_test_key][layer].max(dim=1)[0] for layer in target_modules], dim=1)
            test_data = torch.vstack((test_pos, test_neg))

            if neg_test_key in self._scalers:
                test_data = self._scalers[neg_test_key].transform(test_data)

            y_test = lr.predict_proba(test_data)[:, 1]

            scores_pos = torch.tensor(y_test)[:len(test_pos)].reshape(-1)
            scores_neg = torch.tensor(y_test)[len(test_pos):].reshape(-1)
            self._record(ds_key=pos_test_key, scores=scores_pos, calib_key=neg_test_key)
            self._record(ds_key=neg_test_key, scores=scores_neg, calib_key=neg_test_key)

        return self._df

    def _save_fitting(self, **kwargs):
        self.path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'lrs': self._lrs,
            'scalers': self._scalers,
            }, self._fit_file)
        return

    def load(self, **kwargs):
        if not self._fit_file.exists():
            return 0

        fitting = torch.load(self._fit_file, weights_only=False)
        self._lrs = fitting['lrs']
        self._scalers = fitting['scalers']
        self._fitted = True
        return 1

class DMDScoreConf(Score):
    '''
    Compute DMD-based confidence scores with a linear regressor trained on a balanced subset of correctly and miss-classified samples of a single loader. The features are the maximum activation over the peepholes of each target module. `fit()` must be called once before scoring.

    Args:
    - peepholes (peepholelib.peepholes.peepholes.Peepholes): peepholes from which the features are extracted.
    - loaders (list[str]): loaders on which to compute the scores.
    - target_modules (list[str]): list of target modules, as keys from the model `state_dict`. Should be the same ones passed to `fit()`.
    - verbose (bool): print progress messages.
    '''

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'DMD-in')
        Score.__init__(self, **kwargs)

        # computed in fit()
        self._lr = None
        self._scaler = None
        return

    def fit(self, **kwargs):
        '''
        Train the logistic regressor on a balanced subset of correctly and incorrectly classified samples of `fit_key`.

        Args:
        - peepholes (peepholelib.peepholes.peepholes.Peepholes): peepholes from which the features are extracted.
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed datasets, providing the `result_key`.
        - fit_key (str): loader used to define positive/negative training samples. Defaults to `'train'`.
        - target_modules (list[str]): list of target modules, as keys from the model `state_dict`. If `None`, uses all modules in `peepholes._phs[fit_key]`. Defaults to `None`.
        - scaling (bool): standardize the features before fitting the regressor. Defaults to `False`.
        - result_key (str): key used to read the correct classification flag from the dataset. Defaults to `'result'`.
        - verbose (bool): print progress messages.
        '''
        phs = kwargs['peepholes']
        dss = kwargs['datasets']
        fit_key = kwargs.get('fit_key', 'train')
        target_modules = kwargs.get('target_modules') or list(phs._phs[fit_key].keys())
        scaling = kwargs.get('scaling', False)
        result_key = kwargs.get('result_key', 'result')
        verbose = kwargs.get('verbose', False)

        idx_neg = (dss._dss[fit_key][result_key] == 0).argwhere().squeeze(1)
        idx_pos = (dss._dss[fit_key][result_key] == 1).argwhere().squeeze(1)
        idx_pos = idx_pos[torch.randperm(len(idx_pos))[:len(idx_neg)]]

        if verbose: print(f'Fitting {self.name} on {len(idx_pos)} positive and {len(idx_neg)} negative samples of {fit_key}')

        train_pos = torch.stack([phs._phs[fit_key][layer].max(dim=1)[0][idx_pos] for layer in target_modules], dim=1)
        train_neg = torch.stack([phs._phs[fit_key][layer].max(dim=1)[0][idx_neg] for layer in target_modules], dim=1)

        train_data = torch.vstack((train_pos, train_neg))
        train_label = torch.hstack((torch.ones(len(train_pos)), torch.zeros(len(train_neg))))

        if scaling:
            self._scaler = StandardScaler()
            train_data = self._scaler.fit_transform(train_data)

        # You can use torch tensors for the LogisticRegressionCV, no need to numpy
        self._lr = LogisticRegressionCV(n_jobs=-1, max_iter=5000).fit(train_data, train_label)
        self._save_fitting()
        return

    def compute(self, **kwargs):
        if self._lr is None:
            raise RuntimeError(f'{self.name} regressor not computed. Please run fit() first.')

        phs = kwargs['peepholes']
        loaders = kwargs.get('loaders') or list(phs._phs.keys())
        target_modules = kwargs.get('target_modules') or list(phs._phs[loaders[0]].keys())
        verbose = kwargs.get('verbose', False)

        # skip the loaders already computed
        loaders = [k for k in loaders if not self._is_computed(ds_key=k)]

        for ds_key in loaders:
            if verbose: print('Computing', self.name, 'for dataset', ds_key)

            test_data = torch.stack([phs._phs[ds_key][layer].max(dim=1)[0] for layer in target_modules], dim=1)

            if self._scaler is not None:
                test_data = self._scaler.transform(test_data)

            scores = torch.tensor(self._lr.predict_proba(test_data)[:, 1]).reshape(-1)
            self._record(ds_key=ds_key, scores=scores)

        return self._df

    def _save_fitting(self, **kwargs):
        self.path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'lr': self._lr,
            'scaler': self._scaler,
            }, self._fit_file)
        return

    def load(self, **kwargs):
        if not self._fit_file.exists():
            return 0

        fitting = torch.load(self._fit_file, weights_only=False)
        self._lr = fitting['lr']
        self._scaler = fitting['scaler']
        return 1
