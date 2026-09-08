import torch
from torch.nn.functional import softmax as sm
from peepholelib.scores.score import Score


class ProtoClassScore(Score):
    '''
    Compute the Proto-Class score of all conceptograms in `peepholes._phs[<loaders>]`. `target_modules` are passed to `peepholes.get_conceptograms()` so the evaluation only considers the indicated modules. The score is computed by comparing the conceptogram with the protoclasses. `fit()` must be called once before scoring.

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): datasets respective to the `peepholes`.
    - peepholes (peepholelib.peepholes.peepholes.Peepholes): peepholes from which to take the conceptograms.
    - loaders (list[str]): loaders to consider. If `None`, gets all loaders in `peepholes._phs`. Defaults to `None`.
    - target_modules (list[str]): list of target modules, as keys from the model `state_dict`. Should be the same ones passed to `fit()`.
    - prediction_key (str): key used to read the model's predicted class from the dataset. Defaults to `'pred'`.
    - verbose (bool): print progress messages.
    '''

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'Proto-Class')
        Score.__init__(self, **kwargs)

        # computed in fit()
        # proto: (n_classes, n_modules, n_classes), each element in the first dim
        # is the protoclass of the respective label
        self.proto = None
        return

    def fit(self, **kwargs):
        '''
        Compute the protoclasses from the correctly classified samples of `fit_key` whose model confidence is above `proto_threshold`.

        Args:
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): datasets respective to the `peepholes`.
        - peepholes (peepholelib.peepholes.peepholes.Peepholes): peepholes from which to take the conceptograms.
        - fit_key (str): loader used to compute the protoclasses. Defaults to `'train'`.
        - target_modules (list[str]): list of target modules, as keys from the model `state_dict`.
        - proto_threshold (float): model's confidence threshold to select samples for the protoclass computation (`0 <= proto_threshold <= 1`). Defaults to 0.9.
        - proto (torch.Tensor): protoclasses of shape `(n_classes, n_modules, n_classes)`. If given, they are used as they are instead of being computed. Defaults to `None`.
        - label_key (str): key used to read the class label from the dataset. Defaults to `'label'`.
        - result_key (str): key used to read the correct classification flag from the dataset. Defaults to `'result'`.
        - output_key (str): key used to read the model outputs from the dataset. Defaults to `'output'`.
        - verbose (bool): print progress messages.
        '''
        dss = kwargs['datasets']
        phs = kwargs['peepholes']
        fit_key = kwargs.get('fit_key', 'train')
        target_modules = kwargs.get('target_modules', None)
        proto_th = kwargs.get('proto_threshold', 0.9)
        proto = kwargs.get('proto', None)
        label_key = kwargs.get('label_key', 'label')
        result_key = kwargs.get('result_key', 'result')
        output_key = kwargs.get('output_key', 'output')
        verbose = kwargs.get('verbose', False)

        if proto is not None:
            self.proto = proto
            self._save_fitting()
            return

        if verbose: print('Fitting', self.name, 'on dataset', fit_key)

        cps = phs.get_conceptograms(
                loaders = [fit_key],
                target_modules = target_modules,
                verbose = verbose
                )[fit_key]

        n_modules = cps.shape[1] # number of layers (distributions)
        n_classes = cps.shape[2] # number of classes

        labels = dss._dss[fit_key][:][label_key]
        results = dss._dss[fit_key][:][result_key]
        confs = sm(dss._dss[fit_key][:][output_key], dim=-1).max(dim=-1).values

        proto = torch.zeros(n_classes, n_modules, n_classes)
        for i in range(n_classes):
            idx = (labels == i) & (results == 1) & (confs > proto_th)

            if idx.sum() == 0:
                raise RuntimeError(f'No correctly classified samples with confidence > {proto_th} in "{fit_key}" for class {i}. Consider lowering `proto_threshold`.')

            _p = cps[idx].sum(dim=0)  ## P'_j
            _p /= _p.sum(dim=1, keepdim=True)
            proto[i] = _p

        self.proto = proto

        self._save_fitting()
        return

    def _compute(self, **kwargs):
        if self.proto is None:
            raise RuntimeError(f'{self.name} protoclasses not computed. Please run fit() first.')

        dss = kwargs['datasets']
        phs = kwargs['peepholes']
        loaders = kwargs.get('loaders') or list(phs._phs.keys())
        target_modules = kwargs.get('target_modules', None)
        prediction_key = kwargs.get('prediction_key', 'pred')
        verbose = kwargs.get('verbose', False)

        # get conceptograms
        cpss = phs.get_conceptograms(
                loaders = loaders,
                target_modules = target_modules,
                verbose = verbose
                )

        for ds_key in loaders:
            if self._is_computed(ds_key=ds_key):
                if verbose: print(ds_key, self.name, 'already computed, skipping')
                continue

            if verbose: print('Computing', self.name, 'for dataset', ds_key)

            cps = cpss[ds_key]
            pred = dss._dss[ds_key][prediction_key].int()

            _scores = (self.proto[pred]*cps).sum(dim=(1, 2))
            norm_proto = self.proto[pred].norm(dim=(1, 2))
            norm_cps = cps.norm(dim=(1, 2))

            scores = (_scores/(norm_proto*norm_cps)).detach().cpu().reshape(-1)
            self._record(ds_key=ds_key, scores=scores)

        return self._df

    def _save_fitting(self, **kwargs):
        self.path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'proto': self.proto,
            }, self._fit_file)
        return

    def load(self, **kwargs):
        if not self._fit_file.exists():
            return 0

        fitting = torch.load(self._fit_file, weights_only=False)
        self.proto = fitting['proto']
        return 1
