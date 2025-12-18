from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from peepholelib.scores.score import Score


def _vp_noise_schedule(t, beta_min=0.1, beta_max=20.0):
    integral = beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2
    gamma_t = torch.exp(-0.5 * integral)
    sigma_t = (1.0 - torch.exp(-integral)).sqrt()
    return gamma_t, sigma_t


def _gaussian_kernel(a, b, sigma):
    a_sq = (a ** 2).sum(1, keepdim=True)
    b_sq = (b ** 2).sum(1, keepdim=True)
    sq_dist = (a_sq + b_sq.t() - 2.0 * torch.mm(a, b.t())).clamp(min=0.0)
    return torch.exp(-sq_dist / (2.0 * sigma ** 2))


def _compute_eps(images, score_fn, T_star, beta_min, beta_max, device):
    B = images.shape[0]
    accum = torch.zeros_like(images)
    timesteps = torch.linspace(1.0 / T_star, 1.0, T_star, device=device)
    for t_val in timesteps:
        t = t_val.expand(B)
        gamma_t, sigma_t = _vp_noise_schedule(t, beta_min, beta_max)
        view = (-1,) + (1,) * (images.dim() - 1)
        x_t = gamma_t.view(*view) * images + sigma_t.view(*view) * torch.randn_like(images)
        with torch.no_grad():
            accum += score_fn(x_t, t)
    return (accum / T_star).flatten(1)


def EPS_fit(**kwargs):
    '''
    Compute and save reference EPS vectors from clean (in-distribution) samples.
    Must be called once before EPSScore.

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - score_fn (callable): diffusion score function (x_t, t) -> score.
    - fit_key (str): loader key for reference samples. Defaults to 'train'.
    - stats_path (str | Path): where to save reference EPSs (.pt).
    - T_star (int): number of noise levels. Defaults to 20.
    - beta_min (float): Defaults to 0.1.
    - beta_max (float): Defaults to 20.0.
    - batch_size (int): Defaults to 64.
    - n_threads (int): Defaults to 1.
    - device (str | torch.device): Defaults to 'cpu'.
    - verbose (bool): Defaults to False.
    '''
    datasets = kwargs['datasets']
    score_fn = kwargs['score_fn']
    fit_key = kwargs.get('fit_key', 'train')
    stats_path = Path(kwargs['stats_path'])
    T_star = kwargs.get('T_star', 20)
    beta_min = kwargs.get('beta_min', 0.1)
    beta_max = kwargs.get('beta_max', 20.0)
    bs = kwargs.get('batch_size', 64)
    n_threads = kwargs.get('n_threads', 1)
    device = kwargs.get('device', 'cpu')
    verbose = kwargs.get('verbose', False)

    if stats_path.exists():
        if verbose:
            print(f'EPS reference stats already exist at {stats_path}. Skipping fit.')
        return

    dl = DataLoader(datasets._dss[fit_key], batch_size=bs, shuffle=False,
                    collate_fn=lambda x: x, num_workers=n_threads,
                    pin_memory=(str(device) != 'cpu'))

    all_eps = []
    for batch in tqdm(dl, disable=not verbose, desc='EPS fit'):
        images = batch['image'].to(device)
        eps = _compute_eps(images, score_fn, T_star, beta_min, beta_max, device)
        all_eps.append(eps.detach().cpu())

    ref_eps = torch.cat(all_eps, dim=0)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'ref_eps': ref_eps, 'T_star': T_star,
                'beta_min': beta_min, 'beta_max': beta_max}, stats_path)

    if verbose:
        print(f'EPS reference saved to {stats_path} ({len(ref_eps)} samples).')


class EPSScore(Score):
    score_name = 'EPS'

    def _compute(self, **kwargs):
        datasets = kwargs['datasets']
        score_fn = kwargs['score_fn']
        loaders = kwargs.get('loaders') or list(datasets._dss.keys())
        stats_path = Path(kwargs['stats_path'])
        mmd_sigma = kwargs.get('mmd_sigma', 1.0)
        ref_chunk = kwargs.get('ref_chunk_size', 512)
        bs = kwargs.get('batch_size', 64)
        n_threads = kwargs.get('n_threads', 1)
        device = kwargs.get('device', 'cpu')
        score_name = kwargs.get('score_name', self.score_name)
        verbose = kwargs.get('verbose', False)

        stats = torch.load(stats_path, map_location=device, weights_only=True)
        ref_eps = stats['ref_eps'].to(device)
        T_star = stats['T_star']
        beta_min = stats['beta_min']
        beta_max = stats['beta_max']
        n_ref = ref_eps.shape[0]

        ref_ref_sum = 0.0
        for i in range(0, n_ref, ref_chunk):
            for j in range(0, n_ref, ref_chunk):
                ref_ref_sum += _gaussian_kernel(
                    ref_eps[i:i + ref_chunk], ref_eps[j:j + ref_chunk], mmd_sigma
                ).sum().item()
        ref_ref_term = ref_ref_sum / (n_ref ** 2)

        for ds_key in loaders:
            if self._is_computed(ds_key, score_name):
                if verbose:
                    print(ds_key, score_name, 'already computed, skipping')
                continue
            if verbose:
                print('Computing', score_name, 'for dataset', ds_key)

            dl = DataLoader(datasets._dss[ds_key], batch_size=bs, shuffle=False,
                            collate_fn=lambda x: x, num_workers=n_threads)

            all_scores = []
            for batch in tqdm(dl, disable=not verbose, desc=f'EPS [{ds_key}]'):
                images = batch['image'].to(device)
                test_eps = _compute_eps(images, score_fn, T_star, beta_min, beta_max, device)
                B = test_eps.shape[0]
                cross_sum = torch.zeros(B, device=device)
                for i in range(0, n_ref, ref_chunk):
                    cross_sum += _gaussian_kernel(
                        test_eps, ref_eps[i:i + ref_chunk], mmd_sigma).sum(dim=1)
                mmd_scores = ref_ref_term + (-2.0 * cross_sum / n_ref) + 1.0
                all_scores.append(mmd_scores.detach().cpu())

            self._record(ds_key, torch.cat(all_scores).reshape(-1), score_name)

        return self._df
