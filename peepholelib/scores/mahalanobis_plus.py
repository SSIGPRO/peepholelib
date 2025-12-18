from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from peepholelib.scores.score import Score


def _find_classification_head(model_wrap):
    last_name = None
    for name, module in model_wrap._model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
    if last_name is None:
        raise RuntimeError('No nn.Linear module found in the model.')
    return last_name


def MahalanobisPlus_fit(**kwargs):
    '''
    Fit class-conditional means and shared precision matrix on L2-normalized
    penultimate-layer features. Identical to DMD_base_fit except features are
    L2-normalized before all statistics are computed.
    Reference: https://arxiv.org/abs/2505.18032

    Args:
    - model (peepholelib.models.model_wrap.ModelWrap): wrapped model.
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - fit_key (str): loader key used for fitting. Defaults to 'train'.
    - n_classes (int): number of classes.
    - stats_path (str | Path): path where the statistics .pt file will be saved.
    - batch_size (int): Defaults to 128.
    - n_threads (int): number of dataloader workers. Defaults to 1.
    - verbose (bool): Defaults to False.
    '''
    model = kwargs['model']
    datasets = kwargs['datasets']
    fit_key = kwargs.get('fit_key', 'train')
    n_classes = kwargs['n_classes']
    stats_path = Path(kwargs['stats_path'])
    bs = kwargs.get('batch_size', 128)
    n_threads = kwargs.get('n_threads', 1)
    verbose = kwargs.get('verbose', False)

    if stats_path.exists():
        if verbose:
            print(f'Maha++ statistics already exist at {stats_path}. Skipping fit.')
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

    n_samples = len(datasets._dss[fit_key])
    with torch.no_grad():
        sample = datasets._dss[fit_key][0:1]
        _ = model(sample['image'].to(device))
        act0 = model._acts['in_activations'][layer]
        n_features = act0.view(act0.shape[0], -1).shape[1]

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'{stats_path.stem}_tmp_', dir=stats_path.parent) as tmp_dir:
        tmp_file = Path(tmp_dir) / 'activations'
        tmp_acts = PersistentTensorDict(filename=tmp_file, batch_size=[n_samples], mode='w')
        tmp_acts['acts'] = MMT.empty(shape=(n_samples, n_features), dtype=torch.float32)
        tmp_acts.close()
        tmp_acts = PersistentTensorDict.from_h5(tmp_file, mode='r+')

        try:
            all_labels = []
            counts = torch.zeros(n_classes, dtype=torch.long)
            sums = torch.zeros(n_classes, n_features, dtype=torch.float64)
            offset = 0
            for batch in tqdm(dl, disable=not verbose, total=ceil(n_samples / bs), desc='Maha++ fit'):
                inputs = batch['image'].to(device)
                labels = batch['label'].long()
                all_labels.append(labels)

                with torch.no_grad():
                    _ = model(inputs)

                acts = model._acts['in_activations'][layer]
                acts = acts.view(acts.shape[0], -1).detach().cpu().float()
                acts = F.normalize(acts, p=2, dim=1)
                n = acts.shape[0]
                tmp_acts['acts'][offset:offset + n] = acts
                offset += n

                sums.index_add_(0, labels, acts.double())
                counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.long))

            all_labels = torch.cat(all_labels)

            means = torch.zeros(n_classes, n_features)
            non_empty = counts > 0
            means[non_empty] = (sums[non_empty] / counts[non_empty].unsqueeze(1)).float()

            covariance_matrix = torch.zeros(n_features, n_features, dtype=torch.float64)
            cov_dl = DataLoader(tmp_acts, batch_size=bs, shuffle=False,
                                collate_fn=lambda x: x, num_workers=n_threads)
            for i, batch in enumerate(tqdm(cov_dl, disable=not verbose, desc='Maha++ covariance')):
                acts = batch['acts'].double()
                labels = all_labels[i * bs: i * bs + acts.shape[0]]
                centered = acts - means[labels].double()
                covariance_matrix.addmm_(centered.t(), centered)

            covariance_matrix /= n_samples
            try:
                precision = torch.linalg.pinv(covariance_matrix, hermitian=True).float()
            except TypeError:
                precision = torch.linalg.pinv(covariance_matrix).float()
        finally:
            tmp_acts.close()

    torch.save({'means': means, 'precision': precision}, stats_path)

    if verbose:
        print(f'Maha++ statistics saved to {stats_path}')


class MahalanobisPlusScore(Score):
    score_name = 'Maha++'

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
        layer = _find_classification_head(model)

        stats = torch.load(stats_path, map_location=device, weights_only=True)
        means = stats['means']
        precision = stats['precision']
        n_classes = means.shape[0]

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
            for batch in tqdm(dl, disable=not verbose, desc=f'Maha++ [{ds_key}]'):
                inputs = batch['image'].to(device)
                with torch.no_grad():
                    _ = model(inputs)
                acts = model._acts['in_activations'][layer]
                acts = acts.view(acts.shape[0], -1)
                acts = F.normalize(acts, p=2, dim=1)
                n = acts.shape[0]
                gaussian_score = torch.zeros(n, n_classes, device=device)
                for c in range(n_classes):
                    zero_f = acts - means[c]
                    term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                    gaussian_score[:, c] = term_gau
                ds_scores[write_ptr:write_ptr + n] = gaussian_score.max(dim=1)[0].detach().cpu()
                write_ptr += n

            self._record(ds_key, ds_scores, score_name)

        return self._df
