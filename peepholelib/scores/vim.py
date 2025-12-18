from math import ceil
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from peepholelib.scores.score import Score


def _find_classification_head(model_wrap):
    last_name = None
    for name, module in model_wrap._model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
    if last_name is None:
        raise RuntimeError('No nn.Linear module found in the model.')
    return last_name


def VIM_fit(**kwargs):
    '''
    Fit ViM statistics: classifier null point u, principal subspace U, and alpha.

    u = -pinv(W) @ b is the point in feature space that maps to zero logits,
    used to centre features before covariance estimation (following OpenOOD).
    The null space of the top-d principal components captures directions of
    low variance; a large projection onto it signals OOD.

    Reference: Wang et al., "ViM: Out-Of-Distribution with Virtual-logit Matching", CVPR 2022.
               https://arxiv.org/abs/2203.10807

    Args:
    - model (peepholelib.models.model_wrap.ModelWrap): wrapped model.
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - fit_key (str): loader key for training data.  Defaults to 'train'.
    - stats_path (str | Path): where to save the .pt statistics file.
    - principal_dim (int): number of principal components to keep (d).
                           Defaults to min(feature_dim - 1, 1000).
    - batch_size (int): Defaults to 128.
    - n_threads (int): dataloader workers.  Defaults to 1.
    - verbose (bool): Defaults to False.
    '''
    model = kwargs['model']
    datasets = kwargs['datasets']
    fit_key = kwargs.get('fit_key', 'train')
    stats_path = Path(kwargs['stats_path'])
    principal_dim = kwargs.get('principal_dim', None)
    bs = kwargs.get('batch_size', 128)
    n_threads = kwargs.get('n_threads', 1)
    verbose = kwargs.get('verbose', False)

    if stats_path.exists():
        if verbose:
            print(f'VIM statistics already exist at {stats_path}. Skipping fit.')
        return

    device = model.device
    layer = _find_classification_head(model)

    model.set_target_modules(target_modules=[layer])
    model.set_activations(save_input=True, save_output=False)
    model._model.eval()

    dl = DataLoader(
        datasets._dss[fit_key],
        batch_size=bs,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=n_threads,
        pin_memory=(device != 'cpu'),
    )

    N = len(datasets._dss[fit_key])

    with torch.no_grad():
        sample = datasets._dss[fit_key][0:1]
        _ = model(sample['image'].to(device))
        feat0 = model._acts['in_activations'][layer]
        D = feat0.view(1, -1).shape[1]

    linear = model._model.get_submodule(layer)
    W = linear.weight.detach().cpu()
    b = linear.bias.detach().cpu() if linear.bias is not None else torch.zeros(W.shape[0])
    u = (-torch.linalg.pinv(W) @ b).float()

    if principal_dim is None:
        principal_dim = min(D - 1, 1000)
    principal_dim = min(principal_dim, D - 1)

    cov = torch.zeros(D, D, dtype=torch.float64)
    for batch in tqdm(dl, disable=not verbose, total=ceil(N / bs), desc='VIM fit [cov]'):
        inputs = batch['image'].to(device)
        with torch.no_grad():
            _ = model(inputs)
        feat = model._acts['in_activations'][layer].view(inputs.shape[0], -1).detach().cpu()
        centered = (feat - u).double()
        cov.addmm_(centered.T, centered)
    cov /= N

    _, eigvecs = torch.linalg.eigh(cov.float())
    U = eigvecs[:, -principal_dim:].contiguous()

    vlogit_sum = 0.0
    max_logit_sum = 0.0
    for batch in tqdm(dl, disable=not verbose, total=ceil(N / bs), desc='VIM fit [alpha]'):
        inputs = batch['image'].to(device)
        with torch.no_grad():
            logits = model(inputs)
        feat = model._acts['in_activations'][layer].view(inputs.shape[0], -1).detach().cpu()
        centered = feat - u
        proj = centered @ U
        residual_sq = (centered.pow(2).sum(1) - proj.pow(2).sum(1)).clamp(min=0)
        vlogit_sum += residual_sq.sqrt().sum().item()
        max_logit_sum += logits.detach().cpu().max(dim=1).values.sum().item()

    alpha = max_logit_sum / vlogit_sum

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'u': u, 'U': U, 'alpha': alpha}, stats_path)

    if verbose:
        print(f'VIM statistics saved to {stats_path}')


class VIMScore(Score):
    score_name = 'ViM'

    def _compute(self, **kwargs):
        model = kwargs['model']
        datasets = kwargs['datasets']
        loaders = kwargs.get('loaders') or list(datasets._dss.keys())
        stats_path = Path(kwargs['stats_path'])
        bs = kwargs.get('batch_size', 128)
        n_threads = kwargs.get('n_threads', 1)
        score_name = kwargs.get('score_name', self.score_name)
        verbose = kwargs.get('verbose', False)

        device = model.device
        stats = torch.load(stats_path, map_location=device, weights_only=True)
        u = stats['u']
        U = stats['U']
        alpha = stats['alpha']

        layer = _find_classification_head(model)
        model.set_target_modules(target_modules=[layer])
        model.set_activations(save_input=True, save_output=False)
        model._model.eval()

        for ds_key in loaders:
            if self._is_computed(ds_key, score_name):
                if verbose:
                    print(ds_key, score_name, 'already computed, skipping')
                continue
            if verbose:
                print('Computing', score_name, 'for dataset', ds_key)

            dss = datasets._dss[ds_key]
            n_samples = len(dss)
            ds_scores = torch.empty(n_samples)
            dl = DataLoader(dss, batch_size=bs, shuffle=False,
                            collate_fn=lambda x: x, num_workers=n_threads)

            write_ptr = 0
            for batch in tqdm(dl, disable=not verbose, desc=f'ViM [{ds_key}]'):
                inputs = batch['image'].to(device)
                with torch.no_grad():
                    logits = model(inputs)
                feat = model._acts['in_activations'][layer].view(inputs.shape[0], -1)
                centered = feat - u
                proj = centered @ U
                residual_sq = (centered.pow(2).sum(1) - proj.pow(2).sum(1)).clamp(min=0)
                vlogit = alpha * residual_sq.sqrt()
                extended = torch.cat([logits, vlogit.unsqueeze(1)], dim=1)
                scores = (1 - torch.softmax(extended, dim=-1)[:, -1]).detach().cpu()
                B = inputs.shape[0]
                ds_scores[write_ptr:write_ptr + B] = scores
                write_ptr += B

            self._record(ds_key, ds_scores, score_name)

        return self._df
