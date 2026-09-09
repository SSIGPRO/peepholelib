import torch
from torch.nn.functional import softmax as sm
from peepholelib.scores.score import Score

class ModelOutputScore(Score):
    '''
    Compute a score which is a function of the model outputs only (`datasets._dss[<loaders>][output_key]`). The `type` selects which function is applied to the outputs, among `self.score_fns`:
    - `'MSP'`: maximum softmax probability.
    - `'MaxLogit'`: maximum logit.
    - `'Energy'`: energy score, takes a `temperature`.
    - `'PE'`: normalized and inverted predictive entropy.

    Args:
    - type (str): key of the score function to apply. Also the default `name`.
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - loaders (list[str]): loaders to consider. If `None`, gets all loaders in `datasets._dss`. Defaults to `None`.
    - temperature (float): temperature factor, only used by `'Energy'`. Defaults to 1.0.
    - output_key (str): key used to read the model outputs from the dataset. Defaults to `'output'`.
    - verbose (bool): print progress messages.
    '''

    def __init__(self, **kwargs):
        _type = kwargs['type']

        # score functions available, keyed by type
        self.score_fns = {
                'MSP': self.MSP,
                'MaxLogit': self.max_logit,
                'Energy': self.energy,
                'PE': self.predictive_entropy,
                }

        if _type not in self.score_fns:
            raise ValueError(f'Unknown score type "{_type}". Available types: {list(self.score_fns.keys())}')

        kwargs.setdefault('name', _type)
        Score.__init__(self, **{k: v for k, v in kwargs.items() if k != 'type'})

        self.type = _type
        self.score_fn = self.score_fns[_type]
        return

    def MSP(self, **kwargs):
        '''
        Maximum softmax probability, `max(softmax(model output))`.

        Args:
        - output (torch.Tensor): model outputs, of shape `(n_samples, n_classes)`.
        '''
        output = kwargs['output']

        return sm(output, dim=-1).max(dim=-1).values

    def max_logit(self, **kwargs):
        '''
        Maximum logit, `max(model output)`.

        Args:
        - output (torch.Tensor): model outputs, of shape `(n_samples, n_classes)`.
        '''
        output = kwargs['output']

        return output.max(dim=-1).values

    def energy(self, **kwargs):
        '''
        Energy score, `T*logsumexp(model output/T)`.

        Args:
        - output (torch.Tensor): model outputs, of shape `(n_samples, n_classes)`.
        - temperature (float): temperature factor `T`. Defaults to 1.0.
        '''
        output = kwargs['output']
        temperature = kwargs.get('temperature', 1.0)

        return temperature*(output/temperature).logsumexp(dim=-1)

    def predictive_entropy(self, **kwargs):
        '''
        Predictive entropy, normalized by `log(n_classes)` and inverted, higher values indicate more confident samples.

        Args:
        - output (torch.Tensor): model outputs, of shape `(n_samples, n_classes)`.
        '''
        output = kwargs['output']

        n_classes = output.shape[-1]
        probs = sm(output, dim=-1)
        entropy = -(probs*probs.clamp(min=1e-8).log()).sum(dim=-1)

        return 1.0 - entropy/torch.tensor(n_classes, dtype=torch.float).log()

    def compute(self, **kwargs):
        dss = kwargs['datasets']
        loaders = kwargs.get('loaders') or list(dss._dss.keys())
        temperature = kwargs.get('temperature', 1.0)
        output_key = kwargs.get('output_key', 'output')
        verbose = kwargs.get('verbose', False)

        # skip the loaders already computed
        loaders = [k for k in loaders if not self._is_computed(ds_key=k)]

        for ds_key in loaders:
            if verbose: print('Computing', self.name, 'for dataset', ds_key)

            output = dss._dss[ds_key][output_key]

            _scores = self.score_fn(output=output, temperature=temperature)
            scores = _scores.detach().cpu().reshape(-1)
            self._record(ds_key=ds_key, scores=scores)

        return self._df

    def _save_fitting(self, **kwargs):
        return

    def load(self, **kwargs):
        return 1
