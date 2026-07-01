# torch stuff
import torch
from torch.nn.functional import softmax as sm

from peepholelib.scores.score import Score

class RelUScore(Score):
    score_name = 'Rel-U'

    def _compute(self, **kwargs):
        dss = kwargs['datasets']
        loaders = kwargs.get('loaders') or list(dss._dss.keys())
        fit_key = kwargs.get('fit_key', 'train')
        lbd = kwargs.get('lbd', 0.5)
        temperature = kwargs.get('temperature', 1.0)
        score_name = kwargs.get('score_name', self.score_name)
        verbose = kwargs.get('verbose', False)

        results = dss._dss[fit_key]['result']
        outputs = sm(dss._dss[fit_key]['output'] / temperature, dim=-1)
        train_probs_pos = outputs[results == 1]
        train_probs_neg = outputs[results == 0]

        d_pos = torch.einsum('ij,ik->ijk', train_probs_pos, train_probs_pos).mean(dim=0)
        d_neg = torch.einsum('ij,ik->ijk', train_probs_neg, train_probs_neg).mean(dim=0)
        params = -(1 - lbd) * d_pos + lbd * d_neg
        params = torch.tril(params, diagonal=-1)
        params = params + params.T
        params = torch.relu(params)
        params = params / params.norm()

        _scores = torch.diag(outputs @ params @ outputs.T)
        s_min = _scores.min()
        s_max = _scores.max()

        for ds_key in loaders:
            if self._is_computed(ds_key, score_name):
                if verbose:
                    print(ds_key, score_name, 'already computed, skipping')
                continue
            if verbose:
                print('Computing', score_name, 'for dataset', ds_key)
            outputs_ds = sm(dss._dss[ds_key]['output'] / temperature, dim=-1)
            _params = torch.tril(params, diagonal=-1)
            _params = _params + _params.T
            _params = _params / _params.norm()
            scores = torch.diag(outputs_ds @ _params @ outputs_ds.T)
            scores = (1 - ((scores - s_min) / (s_max - s_min)).clip(0.0, 1.0)).detach().cpu().reshape(-1)
            self._record(ds_key, scores, score_name)

        return self._df