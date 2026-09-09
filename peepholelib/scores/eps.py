from math import ceil
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from peepholelib.scores.score import Score


class EPSScore(Score):
    '''
    Compute the EPS score, the MMD between the expected diffusion score of a sample and the one of a set of reference (in-distribution) samples. `fit()` must be called once before scoring.

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - score_fn (callable): diffusion score function `(x_t, t) -> score`.
    - loaders (list[str]): loaders to consider. If `None`, gets all loaders in `datasets._dss`. Defaults to `None`.
    - mmd_sigma (float): bandwidth of the gaussian kernel. Defaults to 1.0.
    - ref_chunk_size (int): chunk size used to accumulate the kernel over the reference EPSs. Defaults to 512.
    - batch_size (int): Defaults to 64.
    - n_threads (int): dataloader workers. Defaults to 1.
    - device (str|torch.device): Defaults to 'cpu'.
    - input_key (str): key used to read the input images from the dataset. Defaults to `'image'`.
    - verbose (bool): print progress messages.
    '''

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'EPS')
        Score.__init__(self, **kwargs)

        # computed in fit()
        self._ref_eps = None
        self._T_star = None
        self._beta_min = None
        self._beta_max = None
        return

    def vp_noise_schedule(self, **kwargs):
        '''
        Variance-preserving noise schedule.

        Args:
        - t (torch.Tensor): timesteps.
        - beta_min (float): Defaults to 0.1.
        - beta_max (float): Defaults to 20.0.
        '''
        t = kwargs['t']
        beta_min = kwargs.get('beta_min', 0.1)
        beta_max = kwargs.get('beta_max', 20.0)

        integral = beta_min*t + 0.5*(beta_max - beta_min)*t**2
        gamma_t = (-0.5*integral).exp()
        sigma_t = (1.0 - (-integral).exp()).sqrt()

        return gamma_t, sigma_t

    def gaussian_kernel(self, **kwargs):
        '''
        Gaussian kernel between every pair of rows of `a` and `b`.

        Args:
        - a (torch.Tensor): of shape `(n_a, n_features)`.
        - b (torch.Tensor): of shape `(n_b, n_features)`.
        - sigma (float): bandwidth.
        '''
        a = kwargs['a']
        b = kwargs['b']
        sigma = kwargs['sigma']

        a_sq = (a**2).sum(1, keepdim=True)
        b_sq = (b**2).sum(1, keepdim=True)
        sq_dist = (a_sq + b_sq.t() - 2.0*(a@b.t())).clamp(min=0.0)

        return (-sq_dist/(2.0*sigma**2)).exp()

    def compute_eps(self, **kwargs):
        '''
        Expected diffusion score of a batch of images, averaged over `T_star` noise levels.

        Args:
        - images (torch.Tensor): batch of images.
        - score_fn (callable): diffusion score function `(x_t, t) -> score`.
        - T_star (int): number of noise levels.
        - beta_min (float): minimum beta of the noise schedule.
        - beta_max (float): maximum beta of the noise schedule.
        - device (str|torch.device): device to perform the computations.
        '''
        images = kwargs['images']
        score_fn = kwargs['score_fn']
        T_star = kwargs['T_star']
        beta_min = kwargs['beta_min']
        beta_max = kwargs['beta_max']
        device = kwargs['device']

        bsz = images.shape[0]
        accum = torch.zeros_like(images)
        timesteps = torch.linspace(1.0/T_star, 1.0, T_star, device=device)

        for t_val in timesteps:
            t = t_val.expand(bsz)
            gamma_t, sigma_t = self.vp_noise_schedule(t=t, beta_min=beta_min, beta_max=beta_max)
            view = (-1,) + (1,)*(images.dim() - 1)
            x_t = gamma_t.view(*view)*images + sigma_t.view(*view)*torch.randn_like(images)
            with torch.no_grad():
                accum += score_fn(x_t, t)

        return (accum/T_star).flatten(1)

    def fit(self, **kwargs):
        '''
        Compute the reference EPS vectors from clean (in-distribution) samples.

        Args:
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
        - score_fn (callable): diffusion score function `(x_t, t) -> score`.
        - fit_key (str): loader key for reference samples. Defaults to `'train'`.
        - T_star (int): number of noise levels. Defaults to 20.
        - beta_min (float): Defaults to 0.1.
        - beta_max (float): Defaults to 20.0.
        - batch_size (int): Defaults to 64.
        - n_threads (int): dataloader workers. Defaults to 1.
        - device (str|torch.device): Defaults to 'cpu'.
        - input_key (str): key used to read the input images from the dataset. Defaults to `'image'`.
        - verbose (bool): print progress messages.
        '''
        dss = kwargs['datasets']
        score_fn = kwargs['score_fn']
        fit_key = kwargs.get('fit_key', 'train')
        T_star = kwargs.get('T_star', 20)
        beta_min = kwargs.get('beta_min', 0.1)
        beta_max = kwargs.get('beta_max', 20.0)
        bs = kwargs.get('batch_size', 64)
        n_threads = kwargs.get('n_threads', 1)
        device = kwargs.get('device', 'cpu')
        input_key = kwargs.get('input_key', 'image')
        verbose = kwargs.get('verbose', False)

        _dss = dss._dss[fit_key]
        n_samples = len(_dss)

        dl = DataLoader(
                _dss,
                batch_size = bs,
                shuffle = False,
                collate_fn = lambda x: x,
                num_workers = n_threads,
                pin_memory = (str(device) != 'cpu')
                )

        ref_eps = []
        for data in tqdm(dl, disable=not verbose, total=ceil(n_samples/bs), desc=f'{self.name} fit'):
            images = data[input_key].to(device)
            _eps = self.compute_eps(
                    images = images,
                    score_fn = score_fn,
                    T_star = T_star,
                    beta_min = beta_min,
                    beta_max = beta_max,
                    device = device
                    )
            ref_eps.append(_eps.detach().cpu())

        self._ref_eps = torch.cat(ref_eps, dim=0)
        self._T_star = T_star
        self._beta_min = beta_min
        self._beta_max = beta_max
        self._save_fitting()

        if verbose: print(f'{self.name} reference computed ({len(self._ref_eps)} samples).')
        return

    def compute(self, **kwargs):
        if self._ref_eps is None:
            raise RuntimeError(f'{self.name} reference not computed. Please run fit() first.')

        dss = kwargs['datasets']
        score_fn = kwargs['score_fn']
        loaders = kwargs.get('loaders') or list(dss._dss.keys())
        mmd_sigma = kwargs.get('mmd_sigma', 1.0)
        ref_chunk = kwargs.get('ref_chunk_size', 512)
        bs = kwargs.get('batch_size', 64)
        n_threads = kwargs.get('n_threads', 1)
        device = kwargs.get('device', 'cpu')
        input_key = kwargs.get('input_key', 'image')
        verbose = kwargs.get('verbose', False)

        # skip the loaders already computed
        loaders = [k for k in loaders if not self._is_computed(ds_key=k)]
        if len(loaders) == 0:
            return self._df

        ref_eps = self._ref_eps.to(device)
        n_ref = ref_eps.shape[0]

        ref_ref_sum = 0.0
        for i in range(0, n_ref, ref_chunk):
            for j in range(0, n_ref, ref_chunk):
                ref_ref_sum += self.gaussian_kernel(
                        a = ref_eps[i:i+ref_chunk],
                        b = ref_eps[j:j+ref_chunk],
                        sigma = mmd_sigma
                        ).sum().item()
        ref_ref_term = ref_ref_sum/(n_ref**2)

        for ds_key in loaders:
            if verbose: print('Computing', self.name, 'for dataset', ds_key)

            _dss = dss._dss[ds_key]
            n_samples = len(_dss)
            cross_sum = torch.empty(n_samples)

            dl = DataLoader(
                    _dss,
                    batch_size = bs,
                    shuffle = False,
                    collate_fn = lambda x: x,
                    num_workers = n_threads
                    )

            write_ptr = 0
            for data in tqdm(dl, disable=not verbose, total=ceil(n_samples/bs), desc=f'{self.name} [{ds_key}]'):
                images = data[input_key].to(device)
                _eps = self.compute_eps(
                        images = images,
                        score_fn = score_fn,
                        T_star = self._T_star,
                        beta_min = self._beta_min,
                        beta_max = self._beta_max,
                        device = device
                        )

                bsz = _eps.shape[0]
                _cross_sum = torch.zeros(bsz, device=device)
                for i in range(0, n_ref, ref_chunk):
                    _cross_sum += self.gaussian_kernel(
                            a = _eps,
                            b = ref_eps[i:i+ref_chunk],
                            sigma = mmd_sigma
                            ).sum(dim=1)

                cross_sum[write_ptr:write_ptr+bsz] = _cross_sum.detach().cpu()
                write_ptr += bsz

            scores = (ref_ref_term - 2.0*cross_sum/n_ref + 1.0).reshape(-1)
            self._record(ds_key=ds_key, scores=scores)

        return self._df

    def _save_fitting(self, **kwargs):
        self.path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'ref_eps': self._ref_eps,
            'T_star': self._T_star,
            'beta_min': self._beta_min,
            'beta_max': self._beta_max,
            }, self._fit_file)
        return

    def load(self, **kwargs):
        if not self._fit_file.exists():
            return 0

        fitting = torch.load(self._fit_file, weights_only=False)
        self._ref_eps = fitting['ref_eps']
        self._T_star = fitting['T_star']
        self._beta_min = fitting['beta_min']
        self._beta_max = fitting['beta_max']
        return 1
