from math import ceil
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from peepholelib.scores.score import Score


class MahalanobisPlusScore(Score):
    '''
    Compute the Mahalanobis++ score on L2-normalized penultimate-layer features. `fit()` must be called once before scoring.

    Reference: https://arxiv.org/abs/2505.18032

    Args:
    - model (peepholelib.models.model_wrap.ModelWrap): wrapped model.
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - output_layer (str): classification head, as a key from the model `state_dict`. Should be the same one passed to `fit()`.
    - loaders (list[str]): loaders to consider. If `None`, gets all loaders in `datasets._dss`. Defaults to `None`.
    - batch_size (int): Defaults to 128.
    - n_threads (int): dataloader workers. Defaults to 1.
    - input_key (str): key used to read the input images from the dataset. Defaults to `'image'`.
    - verbose (bool): print progress messages.
    '''

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'Maha++')
        Score.__init__(self, **kwargs)

        # computed in fit()
        self._means = None
        self._precision = None
        return

    def fit(self, **kwargs):
        '''
        Fit class-conditional means and shared precision matrix. Identical to `peepholelib.scores.dmd.DMDBase.fit()`, except that the features are L2-normalized before all statistics are computed.

        The within-class scatter is accumulated in a single pass as `sum(x*x.T) - sum(n_c*mu_c*mu_c.T)`, so the activations do not need to be kept.

        Args:
        - model (peepholelib.models.model_wrap.ModelWrap): wrapped model.
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
        - output_layer (str): classification head, as a key from the model `state_dict`. Its input activations are used as features.
        - n_classes (int): number of classes.
        - fit_key (str): loader key used for fitting. Defaults to `'train'`.
        - batch_size (int): Defaults to 128.
        - n_threads (int): dataloader workers. Defaults to 1.
        - input_key (str): key used to read the input images from the dataset. Defaults to `'image'`.
        - label_key (str): key used to read the class label from the dataset. Defaults to `'label'`.
        - verbose (bool): print progress messages.
        '''
        model = kwargs['model']
        dss = kwargs['datasets']
        layer = kwargs['output_layer']
        n_classes = kwargs['n_classes']
        fit_key = kwargs.get('fit_key', 'train')
        bs = kwargs.get('batch_size', 128)
        n_threads = kwargs.get('n_threads', 1)
        input_key = kwargs.get('input_key', 'image')
        label_key = kwargs.get('label_key', 'label')
        verbose = kwargs.get('verbose', False)

        device = model.device

        model.set_target_modules(target_modules=[layer])
        model.set_activations(save_input=True, save_output=False)
        model._model.eval()

        _dss = dss._dss[fit_key]
        n_samples = len(_dss)

        dl = DataLoader(
                _dss,
                batch_size = bs,
                shuffle = False,
                collate_fn = lambda x: x,
                num_workers = n_threads,
                pin_memory = (device != 'cpu')
                )

        # dry run to get the number of features
        with torch.no_grad():
            sample = _dss[0:1]
            _ = model(sample[input_key].to(device))
            act0 = model._acts['in_activations'][layer]
            n_features = act0.view(act0.shape[0], -1).shape[1]

        counts = torch.zeros(n_classes, dtype=torch.long)
        sums = torch.zeros(n_classes, n_features, dtype=torch.float64)
        second_moment = torch.zeros(n_features, n_features, dtype=torch.float64)

        for data in tqdm(dl, disable=not verbose, total=ceil(n_samples/bs), desc=f'{self.name} fit'):
            inputs = data[input_key].to(device)
            _labels = data[label_key].long()

            with torch.no_grad():
                _ = model(inputs)

            acts = model._acts['in_activations'][layer]
            acts = acts.view(acts.shape[0], -1).detach().cpu().float()
            acts = (acts/acts.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)).double()

            sums.index_add_(0, _labels, acts)
            counts.index_add_(0, _labels, torch.ones_like(_labels, dtype=torch.long))
            second_moment.addmm_(acts.t(), acts)

        means = torch.zeros(n_classes, n_features)
        non_empty = counts > 0
        means[non_empty] = (sums[non_empty]/counts[non_empty].unsqueeze(1)).float()

        # sum_i (x_i - mu_yi)*(x_i - mu_yi).T = sum_i x_i*x_i.T - sum_c n_c*mu_c*mu_c.T
        _means = means.double()
        covariance_matrix = (second_moment - (counts.unsqueeze(1)*_means).t()@_means)/n_samples

        try:
            precision = torch.linalg.pinv(covariance_matrix, hermitian=True).float()
        except TypeError:
            precision = torch.linalg.pinv(covariance_matrix).float()

        self._means = means
        self._precision = precision
        self._save_fitting()

        # reset the model to NOT get activations
        model.set_activations(save_input=False, save_output=False)
        return

    def compute(self, **kwargs):
        if self._means is None:
            raise RuntimeError(f'{self.name} statistics not computed. Please run fit() first.')

        model = kwargs['model']
        dss = kwargs['datasets']
        layer = kwargs['output_layer']
        loaders = kwargs.get('loaders') or list(dss._dss.keys())
        bs = kwargs.get('batch_size', 128)
        n_threads = kwargs.get('n_threads', 1)
        input_key = kwargs.get('input_key', 'image')
        verbose = kwargs.get('verbose', False)

        # skip the loaders already computed
        loaders = [k for k in loaders if not self._is_computed(ds_key=k)]
        if len(loaders) == 0:
            return self._df

        device = model.device

        means = self._means.to(device)
        precision = self._precision.to(device)
        n_classes = means.shape[0]

        model.set_target_modules(target_modules=[layer])
        model.set_activations(save_input=True, save_output=False)
        model._model.eval()

        for ds_key in loaders:
            if verbose: print('Computing', self.name, 'for dataset', ds_key)

            _dss = dss._dss[ds_key]
            n_samples = len(_dss)
            gaussian_score = torch.empty(n_samples, n_classes)

            dl = DataLoader(
                    _dss,
                    batch_size = bs,
                    shuffle = False,
                    collate_fn = lambda x: x,
                    num_workers = n_threads
                    )

            write_ptr = 0
            for data in tqdm(dl, disable=not verbose, total=ceil(n_samples/bs), desc=f'{self.name} [{ds_key}]'):
                inputs = data[input_key].to(device)
                with torch.no_grad():
                    _ = model(inputs)

                acts = model._acts['in_activations'][layer]
                acts = acts.view(acts.shape[0], -1)
                acts = acts/acts.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)

                bsz = acts.shape[0]
                _gaussian_score = torch.zeros(bsz, n_classes, device=device)
                for c in range(n_classes):
                    zero_f = acts - means[c]
                    _gaussian_score[:, c] = -0.5*(zero_f@precision@zero_f.t()).diag()

                gaussian_score[write_ptr:write_ptr+bsz] = _gaussian_score.detach().cpu()
                write_ptr += bsz

            scores = gaussian_score.max(dim=1)[0].reshape(-1)
            self._record(ds_key=ds_key, scores=scores)

        # reset the model to NOT get activations
        model.set_activations(save_input=False, save_output=False)

        return self._df

    def _save_fitting(self, **kwargs):
        self.path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'means': self._means,
            'precision': self._precision,
            }, self._fit_file)
        return

    def load(self, **kwargs):
        if not self._fit_file.exists():
            return 0

        fitting = torch.load(self._fit_file, weights_only=False)
        self._means = fitting['means']
        self._precision = fitting['precision']
        return 1
