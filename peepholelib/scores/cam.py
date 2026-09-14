import math
from sklearn.metrics import roc_curve

import torch
from peepholelib.scores.score import Score

class CAMLinScore(Score):
    '''
    Compute the CAM linear score of all samples in `peepholes._phs[<loaders>]`. The score is `1 - eta`, with `eta` the cost of the model's predicted class averaged over `target_modules`, higher values indicate better coverage. Assumes the costs lie in [0, 1] (`normalize=True` in the driller).

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed datasets corresponding to `peepholes`. Used to retrieve the model's predicted class for each sample.
    - peepholes (peepholelib.peepholes.peepholes.Peepholes): peepholes containing the MRC cost `eta` for each loader and layer.
    - loaders (list[str]): loaders to consider. If `None`, gets all loaders in `peepholes._phs`. Defaults to `None`.
    - target_modules (list[str]): layers whose `eta` values are averaged. Defaults to all modules in `peepholes` for the first loader.
    - prediction_key (str): key used to read the model's predicted class from the dataset. Defaults to `'pred'`.
    - verbose (bool): print progress messages.
    '''

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'CAM-lin')
        Score.__init__(self, **kwargs)
        return

    def compute(self, **kwargs):
        dss = kwargs['datasets']
        phs = kwargs['peepholes']
        loaders = kwargs.get('loaders') or list(phs._phs.keys())
        target_modules = kwargs.get('target_modules') or list(phs._phs[loaders[0]].keys())
        prediction_key = kwargs.get('prediction_key', 'pred')
        verbose = kwargs.get('verbose', False)

        # skip the loaders already computed
        loaders = [k for k in loaders if not self._is_computed(ds_key=k)]

        for ds_key in loaders:
            if verbose: print('Computing', self.name, 'for dataset', ds_key)

            h = sum(phs._phs[ds_key][layer] for layer in target_modules)/len(target_modules)
            pred = dss._dss[ds_key][:][prediction_key]

            _scores = 1 - h.gather(1, pred.unsqueeze(1)).squeeze(1)
            scores = _scores.detach().cpu().reshape(-1)
            self._record(ds_key=ds_key, scores=scores)

        return self._df

    def _save_fitting(self, **kwargs):
        return

    def load(self, **kwargs):
        return 1

class CAMExpScore(Score):
    '''
    Compute the CAM confidence score `c` for positive (trusted) and negative samples. For each entry in `neg_loaders`, `tau` is calibrated per class using `pos_train_loader` and all corresponding negative train loaders via a ROC (Youden's J). Samples are balanced between negative positive loaders.

    Since the scores of the positive test samples change for each negative loader used in the calibration, the negative test loader they were calibrated against is recorded in the `'calib key'` column.

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed datasets corresponding to `peepholes`. Used to retrieve the model's predicted class for each sample.
    - peepholes (peepholelib.peepholes.peepholes.Peepholes): peepholes containing the MRC cost `eta` for each loader and layer.
    - pos_test_loader (str): loader with positive (trusted) samples to score. Defaults to `'test'`.
    - target_modules (list[str]): layers whose `eta` values are summed to form `h`. Should be the same ones passed to `fit()`.
    - prediction_key (str): key used to read the model's predicted class from the dataset. Defaults to `'pred'`.
    - verbose (bool): print progress messages.
    '''

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'CAM')
        Score.__init__(self, **kwargs)

        # computed in fit(), keyed by the negative TEST loader they will score
        self._taus = {}

        # set in fit()
        self._fitted = False
        return

    def fit(self, **kwargs):
        '''
        Calibrate one `tau` vector (one threshold per class) for each negative loader against `pos_train_loader`.
        The thresholds are kept in `self._taus`, keyed by the negative TEST loader they will be used to score.

        Pairs whose scores are already in `self.df` are skipped.

        Args:
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed datasets corresponding to `peepholes`.
        - peepholes (peepholelib.peepholes.peepholes.Peepholes): peepholes containing the MRC cost `eta` for each loader and layer.
        - pos_train_loader (str): loader with positive (trusted) samples used to calibrate `tau` per class. Defaults to `'val'`.
        - neg_loaders (dict{str: list[str]}): maps each negative test loader to a list of negative train loaders used to calibrate `tau`.
        - target_modules (list[str]): layers whose `eta` values are summed to form `h`. Defaults to all modules in `peepholes` for `pos_train_loader`.
        - prediction_key (str): key used to read the model's predicted class from the dataset. Defaults to `'pred'`.
        - verbose (bool): print progress messages.
        '''
        dss = kwargs['datasets']
        phs = kwargs['peepholes']
        pos_train_key = kwargs.get('pos_train_loader', 'val')
        neg_keys = kwargs['neg_loaders']
        target_modules = kwargs.get('target_modules') or list(phs._phs[pos_train_key].keys())
        prediction_key = kwargs.get('prediction_key', 'pred')
        verbose = kwargs.get('verbose', False)

        pending_neg = {
                k: v for k, v in neg_keys.items()
                if not self._is_computed(ds_key=k, calib_key=k)
                }
        if len(pending_neg) == 0:
            self._fitted = True
            return
        print(f'{self.name}: {len(pending_neg)}/{len(neg_keys)} negative loaders still to fit: {list(pending_neg.keys())}')

        # accumulate h for the positive train loader once, outside the loop over negative pairs
        h_pos_train = sum(phs._phs[pos_train_key][layer] for layer in target_modules)
        pred_pos_train = dss._dss[pos_train_key][:][prediction_key]
        n_classes = h_pos_train.shape[1]

        for neg_test_key, neg_train_keys in pending_neg.items():
            if verbose: print('Fitting', self.name, 'for dataset', neg_test_key)

            n_neg_keys = len(neg_train_keys)

            h_neg_list = [
                    sum(phs._phs[nl][layer] for layer in target_modules)
                    for nl in neg_train_keys
                    ]
            pred_neg_list = [
                    dss._dss[nl][:][prediction_key]
                    for nl in neg_train_keys
                    ]

            # calibrate tau: Youden's J on the ROC for each class
            tau = torch.zeros(n_classes, device=h_pos_train.device)
            for i in range(n_classes):
                hi_pos = h_pos_train[pred_pos_train == i, i]
                n_pos = hi_pos.shape[0]

                if n_pos == 0:
                    tau[i] = float('nan')
                    continue

                # draw n_pos // n_neg_keys samples from each negative train loader
                n_per_loader = n_pos//n_neg_keys
                hi_neg = []
                skip_class = False
                for h_neg, pred_neg in zip(h_neg_list, pred_neg_list):
                    candidates = h_neg[pred_neg == i, i]
                    if candidates.shape[0] == 0:
                        tau[i] = float('nan')
                        skip_class = True
                        break
                    n_take = min(n_per_loader, candidates.shape[0])
                    idx = torch.randperm(candidates.shape[0], device=candidates.device)[:n_take]
                    hi_neg.append(candidates[idx])
                if skip_class:
                    continue

                hi_neg = torch.cat(hi_neg)

                y_true = torch.cat((torch.zeros_like(hi_pos), torch.ones_like(hi_neg)))
                y_score = torch.cat((hi_pos, hi_neg))
                fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_score.cpu().numpy())
                tau[i] = float(thresholds[(tpr - fpr).argmax()])

            nan_mask = tau.isnan()
            if nan_mask.any():
                tau[nan_mask] = tau[~nan_mask].mean()

            self._taus[neg_test_key] = tau

        self._fitted = True
        self._save_fitting()
        return

    def compute(self, **kwargs):
        if not self._fitted:
            raise RuntimeError(f'{self.name} thresholds not computed. Please run fit() first.')

        dss = kwargs['datasets']
        phs = kwargs['peepholes']
        pos_test_key = kwargs.get('pos_test_loader', 'test')
        target_modules = kwargs.get('target_modules') or list(phs._phs[pos_test_key].keys())
        prediction_key = kwargs.get('prediction_key', 'pred')
        verbose = kwargs.get('verbose', False)

        # accumulate h for the positive test loader once, outside the loop over negative pairs
        h_pos_test = sum(phs._phs[pos_test_key][layer] for layer in target_modules)
        pred_pos_test = dss._dss[pos_test_key][:][prediction_key]
        h_pred_pos_test = h_pos_test.gather(1, pred_pos_test.unsqueeze(1)).squeeze(1)

        for neg_test_key, tau in self._taus.items():
            if self._is_computed(ds_key=neg_test_key, calib_key=neg_test_key):
                if verbose: print(self.name, neg_test_key, 'already computed, skipping')
                continue

            if verbose: print('Computing', self.name, 'for dataset', neg_test_key)

            # score positive test samples
            scores_pos = (-h_pred_pos_test*math.log(2)/tau[pred_pos_test]).exp().detach().cpu().reshape(-1)

            # score negative test samples
            h_neg_test = sum(phs._phs[neg_test_key][layer] for layer in target_modules)
            pred_neg_test = dss._dss[neg_test_key][:][prediction_key]
            h_pred_neg_test = h_neg_test.gather(1, pred_neg_test.unsqueeze(1)).squeeze(1)
            scores_neg = (-h_pred_neg_test*math.log(2)/tau[pred_neg_test]).exp().detach().cpu().reshape(-1)

            self._record(ds_key=pos_test_key, scores=scores_pos, calib_key=neg_test_key)
            self._record(ds_key=neg_test_key, scores=scores_neg, calib_key=neg_test_key)

        return self._df

    def _save_fitting(self, **kwargs):
        self.path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'taus': self._taus,
            }, self._fit_file)
        return

    def load(self, **kwargs):
        if not self._fit_file.exists():
            return 0

        fitting = torch.load(self._fit_file, weights_only=False)
        self._taus = fitting['taus']
        self._fitted = True
        return 1
