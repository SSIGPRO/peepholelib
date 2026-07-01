import torch
from peepholelib.scores.score import Score


class PEScore(Score):
    score_name = 'PE'

    def _compute(self, **kwargs):
        dss = kwargs['datasets']
        loaders = kwargs.get('loaders') or list(dss._dss.keys())
        verbose = kwargs.get('verbose', False)

        for ds_key in loaders:
            if self._is_computed(ds_key):
                if verbose:
                    print(ds_key, self.score_name, 'already computed, skipping')
                continue
            if verbose:
                print('Computing', self.score_name, 'for dataset', ds_key)
            logits = dss._dss[ds_key]['output']
            n_classes = logits.shape[-1]
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)
            scores = 1.0 - entropy / torch.log(torch.tensor(n_classes, dtype=torch.float))
            self._record(ds_key, scores.detach().cpu().reshape(-1))

        return self._df