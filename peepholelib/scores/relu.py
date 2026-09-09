import torch
from torch.nn.functional import softmax as sm
from peepholelib.scores.score import Score

class RelUScore(Score):
    '''
    Compute the Relative Uncertainty score as described in https://arxiv.org/abs/2306.01710. `fit()` must be called once before scoring.

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - loaders (list[str]): loaders to consider. If `None`, gets all loaders in `datasets._dss`. Defaults to `None`.
    - temperature (float): temperature factor. Should be the same one passed to `fit()`. Defaults to 1.0.
    - output_key (str): key used to read the model outputs from the dataset. Defaults to `'output'`.
    - verbose (bool): print progress messages.
    '''

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'Rel-U')
        Score.__init__(self, **kwargs)

        # computed in fit()
        self._params = None
        self._s_min = None
        self._s_max = None
        return

    def fit(self, **kwargs):
        '''
        Fit the parameter matrix on the correctly/miss-classified samples of `fit_key`, and the score range used to normalize.

        Args:
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
        - fit_key (str): loader to fit distributions. Usually `'train'`. Defaults to `'train'`.
        - lbd (float): lambda factor. Defaults to 0.5.
        - temperature (float): temperature factor. Defaults to 1.0.
        - output_key (str): key used to read the model outputs from the dataset. Defaults to `'output'`.
        - result_key (str): key used to read the correct classification flag from the dataset. Defaults to `'result'`.
        - verbose (bool): print progress messages.
        '''
        dss = kwargs['datasets']
        fit_key = kwargs.get('fit_key', 'train')
        lbd = kwargs.get('lbd', 0.5)
        temperature = kwargs.get('temperature', 1.0)
        output_key = kwargs.get('output_key', 'output')
        result_key = kwargs.get('result_key', 'result')
        verbose = kwargs.get('verbose', False)

        if verbose: print('Fitting', self.name, 'on dataset', fit_key)

        results = dss._dss[fit_key][result_key]
        probs = sm(dss._dss[fit_key][output_key]/temperature, dim=-1)

        probs_pos = probs[results == 1]
        probs_neg = probs[results == 0]

        if len(probs_pos) == 0 or len(probs_neg) == 0:
            raise RuntimeError(f'Not enough correctly/miss-classified samples in "{fit_key}", found {len(probs_pos)}/{len(probs_neg)} correctly/miss-classified.')

        d_pos = torch.einsum('ij,ik->ijk', probs_pos, probs_pos).mean(dim=0)
        d_neg = torch.einsum('ij,ik->ijk', probs_neg, probs_neg).mean(dim=0)

        params = -(1 - lbd)*d_pos + lbd*d_neg
        params = params.tril(diagonal=-1)
        params = params + params.T
        params = params.relu()
        params = params/params.norm()

        _scores = (probs@params@probs.T).diag()

        self._params = params
        self._s_min = _scores.min()
        self._s_max = _scores.max()
        self._save_fitting()
        return

    def compute(self, **kwargs):
        if self._params is None:
            raise RuntimeError(f'{self.name} parameters not computed. Please run fit() first.')

        dss = kwargs['datasets']
        loaders = kwargs.get('loaders') or list(dss._dss.keys())
        temperature = kwargs.get('temperature', 1.0)
        output_key = kwargs.get('output_key', 'output')
        verbose = kwargs.get('verbose', False)

        # skip the loaders already computed
        loaders = [k for k in loaders if not self._is_computed(ds_key=k)]

        params = self._params.tril(diagonal=-1)
        params = params + params.T
        params = params/params.norm()

        for ds_key in loaders:
            if verbose: print('Computing', self.name, 'for dataset', ds_key)

            probs = sm(dss._dss[ds_key][output_key]/temperature, dim=-1)

            _scores = (probs@params@probs.T).diag()
            _scores = 1 - ((_scores - self._s_min)/(self._s_max - self._s_min)).clip(0.0, 1.0)
            scores = _scores.detach().cpu().reshape(-1)
            self._record(ds_key=ds_key, scores=scores)

        return self._df

    def _save_fitting(self, **kwargs):
        self.path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'params': self._params,
            's_min': self._s_min,
            's_max': self._s_max,
            }, self._fit_file)
        return

    def load(self, **kwargs):
        if not self._fit_file.exists():
            return 0

        fitting = torch.load(self._fit_file, weights_only=False)
        self._params = fitting['params']
        self._s_min = fitting['s_min']
        self._s_max = fitting['s_max']
        return 1
