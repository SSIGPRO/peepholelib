from math import ceil
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from peepholelib.scores.score import Score


class VIMScore(Score):
    '''
    Compute the ViM score. `fit()` must be called once before scoring.

    Reference: Wang et al., "ViM: Out-Of-Distribution with Virtual-logit Matching", CVPR 2022.
               https://arxiv.org/abs/2203.10807

    Args:
    - model (peepholelib.models.model_wrap.ModelWrap): wrapped model.
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - output_layer (str): classification head, as a key from the model `state_dict`. Should be the same one passed to `fit()`.
    - loaders (list[str]): loaders to consider. If `None`, gets all loaders in `datasets._dss`. Defaults to `None`.
    - batch_size (int): Defaults to 128.
    - n_threads (int): dataloader workers. Defaults to 1.
    - input_key (str): key used to read the input images from the dataset. Defaults to `'image'`.
    - output_key (str): key used to read the model outputs from the dataset, to get the number of classes. Defaults to `'output'`.
    - verbose (bool): print progress messages.
    '''

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'ViM')
        Score.__init__(self, **kwargs)

        # computed in fit(), the classifier null point, principal subspace and scaling factor
        self._u = None
        self._U = None
        self._alpha = None
        return

    def fit(self, **kwargs):
        '''
        Fit the ViM statistics: classifier null point `u`, principal subspace `U`, and `alpha`.

        `u = -pinv(W)@b` maps to zero logits, and is used to centre the features before the covariance estimation (following OpenOOD). The null space of the top-d principal components captures directions of low variance, a large projection onto it signals OOD.

        Args:
        - model (peepholelib.models.model_wrap.ModelWrap): wrapped model.
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
        - output_layer (str): classification head, as a key from the model `state_dict`. Its input activations are used as features.
        - fit_key (str): loader key for training data. Defaults to `'train'`.
        - principal_dim (int): number of principal components to keep (d). Defaults to `min(n_features - 1, 1000)`.
        - batch_size (int): Defaults to 128.
        - n_threads (int): dataloader workers. Defaults to 1.
        - input_key (str): key used to read the input images from the dataset. Defaults to `'image'`.
        - verbose (bool): print progress messages.
        '''
        model = kwargs['model']
        dss = kwargs['datasets']
        layer = kwargs['output_layer']
        fit_key = kwargs.get('fit_key', 'train')
        principal_dim = kwargs.get('principal_dim', None)
        bs = kwargs.get('batch_size', 128)
        n_threads = kwargs.get('n_threads', 1)
        input_key = kwargs.get('input_key', 'image')
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

        linear = model._model.get_submodule(layer)
        W = linear.weight.detach().cpu()
        b = linear.bias.detach().cpu() if linear.bias is not None else torch.zeros(W.shape[0])
        u = (-torch.linalg.pinv(W)@b).float()

        if principal_dim is None:
            principal_dim = min(n_features - 1, 1000)
        principal_dim = min(principal_dim, n_features - 1)

        cov = torch.zeros(n_features, n_features, dtype=torch.float64)
        for data in tqdm(dl, disable=not verbose, total=ceil(n_samples/bs), desc=f'{self.name} fit [cov]'):
            inputs = data[input_key].to(device)
            with torch.no_grad():
                _ = model(inputs)
            acts = model._acts['in_activations'][layer].view(inputs.shape[0], -1).detach().cpu()
            centered = (acts - u).double()
            cov.addmm_(centered.T, centered)
        cov /= n_samples

        _, eigvecs = torch.linalg.eigh(cov.float())
        U = eigvecs[:, -principal_dim:].contiguous()

        vlogits = torch.empty(n_samples)
        max_logits = torch.empty(n_samples)

        write_ptr = 0
        for data in tqdm(dl, disable=not verbose, total=ceil(n_samples/bs), desc=f'{self.name} fit [alpha]'):
            inputs = data[input_key].to(device)
            with torch.no_grad():
                output = model(inputs)
            acts = model._acts['in_activations'][layer].view(inputs.shape[0], -1).detach().cpu()
            centered = acts - u
            proj = centered@U
            residual_sq = (centered.pow(2).sum(1) - proj.pow(2).sum(1)).clamp(min=0)

            bsz = inputs.shape[0]
            vlogits[write_ptr:write_ptr+bsz] = residual_sq.sqrt()
            max_logits[write_ptr:write_ptr+bsz] = output.detach().cpu().max(dim=1).values
            write_ptr += bsz

        self._u = u
        self._U = U
        self._alpha = (max_logits.sum()/vlogits.sum()).item()
        self._save_fitting()

        # reset the model to NOT get activations
        model.set_activations(save_input=False, save_output=False)
        return

    def _compute(self, **kwargs):
        if self._u is None:
            raise RuntimeError(f'{self.name} statistics not computed. Please run fit() first.')

        model = kwargs['model']
        dss = kwargs['datasets']
        layer = kwargs['output_layer']
        loaders = kwargs.get('loaders') or list(dss._dss.keys())
        bs = kwargs.get('batch_size', 128)
        n_threads = kwargs.get('n_threads', 1)
        input_key = kwargs.get('input_key', 'image')
        output_key = kwargs.get('output_key', 'output')
        verbose = kwargs.get('verbose', False)

        device = model.device

        u = self._u.to(device)
        U = self._U.to(device)
        alpha = self._alpha

        model.set_target_modules(target_modules=[layer])
        model.set_activations(save_input=True, save_output=False)
        model._model.eval()

        for ds_key in loaders:
            if self._is_computed(ds_key=ds_key):
                if verbose: print(ds_key, self.name, 'already computed, skipping')
                continue

            if verbose: print('Computing', self.name, 'for dataset', ds_key)

            _dss = dss._dss[ds_key]
            n_samples = len(_dss)
            n_classes = _dss[0:1][output_key].shape[-1]

            logits = torch.empty(n_samples, n_classes)
            vlogits = torch.empty(n_samples)

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
                    _logits = model(inputs)
                acts = model._acts['in_activations'][layer].view(inputs.shape[0], -1)
                centered = acts - u
                proj = centered@U
                residual_sq = (centered.pow(2).sum(1) - proj.pow(2).sum(1)).clamp(min=0)

                bsz = inputs.shape[0]
                logits[write_ptr:write_ptr+bsz] = _logits.detach().cpu()
                vlogits[write_ptr:write_ptr+bsz] = (alpha*residual_sq.sqrt()).detach().cpu()
                write_ptr += bsz

            extended = torch.cat([logits, vlogits.unsqueeze(1)], dim=1)
            scores = (1 - extended.softmax(dim=-1)[:, -1]).reshape(-1)
            self._record(ds_key=ds_key, scores=scores)

        # reset the model to NOT get activations
        model.set_activations(save_input=False, save_output=False)

        return self._df

    def _save_fitting(self, **kwargs):
        self.path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'u': self._u,
            'U': self._U,
            'alpha': self._alpha,
            }, self._fit_file)
        return

    def load(self, **kwargs):
        if not self._fit_file.exists():
            return 0

        fitting = torch.load(self._fit_file, weights_only=False)
        self._u = fitting['u']
        self._U = fitting['U']
        self._alpha = fitting['alpha']
        return 1
