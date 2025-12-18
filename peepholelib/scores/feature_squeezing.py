import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from peepholelib.scores.score import Score


class FeatureSqueezingScore(Score):
    score_name = 'Feature-Squeezing'

    def __call__(self, **kwargs):
        dss = kwargs['datasets']
        loaders = kwargs.get('loaders_ori') or list(dss._dss.keys())
        detector = kwargs['detector']
        bs = kwargs.get('batch_size', 64)
        n_threads = kwargs.get('n_threads', 1)
        score_name = kwargs.get('score_name', self.score_name)
        verbose = kwargs.get('verbose', False)

        loaders = self._get_pending_loaders(loaders, score_name)
        if not loaders:
            return self._df

        for ds_key in loaders:
            if self._is_computed(ds_key, score_name):
                if verbose:
                    print(ds_key, score_name, 'already computed, skipping')
                continue
            if verbose:
                print('Computing', score_name, 'for dataset', ds_key)
            dl = DataLoader(dss._dss[ds_key], batch_size=bs, shuffle=False,
                            collate_fn=lambda x: x, num_workers=n_threads)
            ori_list = []
            for data in tqdm(dl, disable=not verbose):
                ori_list.append(detector(data['image']).detach().cpu())
            scores = ((2 - torch.cat(ori_list, dim=0)) / 2).reshape(-1)
            self._record(ds_key, scores, score_name)

        return self._df
