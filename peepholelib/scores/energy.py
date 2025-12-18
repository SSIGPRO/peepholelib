import torch
from peepholelib.scores.score import Score


class EnergyScore(Score):
    score_name = 'Energy'

    def _compute(self, **kwargs):
        dss = kwargs['datasets']
        loaders = kwargs.get('loaders') or list(dss._dss.keys())
        temperature = kwargs.get('temperature', 1.0)
        verbose = kwargs.get('verbose', False)

        for ds_key in loaders:
            if self._is_computed(ds_key):
                if verbose:
                    print(ds_key, self.score_name, 'already computed, skipping')
                continue
            if verbose:
                print('Computing', self.score_name, 'for dataset', ds_key)
            logits = dss._dss[ds_key]['output']
            scores = temperature * torch.logsumexp(logits / temperature, dim=-1)
            self._record(ds_key, scores.detach().cpu().reshape(-1))

        return self._df
