from math import ceil
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from peepholelib.scores.score import Score


class FeatureSqueezingScore(Score):
    '''
    Compute the Feature Squeezing score (https://arxiv.org/abs/1704.01155).

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - loaders (list[str]): loaders to consider. If `None`, gets all loaders in `datasets._dss`. Defaults to `None`.
    - detector (peepholelib.featureSqueezing.FeatureSqueezingDetector): detector used to analyse the input images.
    - batch_size (int): batch size used to compute the scores. Defaults to 64.
    - n_threads (int): `num_workers` passed to `torch.utils.data.DataLoader`. Defaults to 1.
    - input_key (str): key used to read the input images from the dataset. Defaults to `'image'`.
    - verbose (bool): print progress messages.
    '''

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'Feature-Squeezing')
        Score.__init__(self, **kwargs)
        return

    def compute(self, **kwargs):
        dss = kwargs['datasets']
        loaders = kwargs.get('loaders') or list(dss._dss.keys())
        detector = kwargs['detector']
        bs = kwargs.get('batch_size', 64)
        n_threads = kwargs.get('n_threads', 1)
        input_key = kwargs.get('input_key', 'image')
        verbose = kwargs.get('verbose', False)

        # skip the loaders already computed
        loaders = [k for k in loaders if not self._is_computed(ds_key=k)]

        for ds_key in loaders:
            if verbose: print('Computing', self.name, 'for dataset', ds_key)

            _dss = dss._dss[ds_key]
            n_samples = len(_dss)
            distances = torch.empty(n_samples)

            dl = DataLoader(
                    _dss,
                    batch_size = bs,
                    shuffle = False,
                    collate_fn = lambda x: x,
                    num_workers = n_threads
                    )

            write_ptr = 0
            for data in tqdm(dl, disable=not verbose, total=ceil(n_samples/bs), desc=f'{self.name} [{ds_key}]'):
                _distances = detector(data[input_key]).detach().cpu()
                bsz = _distances.shape[0]
                distances[write_ptr:write_ptr+bsz] = _distances
                write_ptr += bsz

            scores = ((2 - distances)/2).reshape(-1)
            self._record(ds_key=ds_key, scores=scores)

        return self._df

    def _save_fitting(self, **kwargs):
        return

    def load(self, **kwargs):
        return 1
