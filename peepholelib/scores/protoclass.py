# torch stuff
import torch
from torch.nn.functional import softmax as sm

from peepholelib.scores.score import Score

class ProtoClassScore(Score):
    score_name = 'Proto-Class'

    def __init__(self, save_path=None):
        super().__init__(save_path)
        self.proto = None

    def _compute(self, **kwargs):
        dss = kwargs['datasets']
        phs = kwargs['peepholes']
        loaders = kwargs.get('loaders') or list(phs._phs.keys())
        target_modules = kwargs.get('target_modules', None)
        proto_key = kwargs.get('proto_key', 'train')
        proto_th = kwargs.get('proto_threshold', 0.9)
        score_name = kwargs.get('score_name', self.score_name)
        proto = kwargs.get('proto', self.proto)
        verbose = kwargs.get('verbose', False)

        fetch_loaders = loaders if proto_key in loaders else loaders + [proto_key]
        cpss = phs.get_conceptograms(loaders=fetch_loaders, target_modules=target_modules, verbose=verbose)

        nd = cpss[loaders[0]].shape[1]
        nc = cpss[loaders[0]].shape[2]

        if proto is None:
            cps = cpss[proto_key]
            labels = dss._dss[proto_key][:]['label']
            results = dss._dss[proto_key][:]['result']
            confs = sm(dss._dss[proto_key][:]['output'], dim=-1).max(dim=-1).values
            proto = torch.zeros(nc, nd, nc)
            for i in range(nc):
                idx = torch.logical_and(labels == i, results == 1)
                idx = torch.logical_and(idx, confs > proto_th)
                _p = cps[idx].sum(dim=0)
                _p /= _p.sum(dim=1, keepdim=True)
                proto[i] = _p
        self.proto = proto

        for ds_key in loaders:
            if self._is_computed(ds_key, score_name):
                if verbose:
                    print(ds_key, score_name, 'already computed, skipping')
                continue
            if verbose:
                print('Computing', score_name, 'for dataset', ds_key)
            cps = cpss[ds_key]
            pred = dss._dss[ds_key]['pred'].int()
            scores = (proto[pred] * cps).sum(dim=(1, 2))
            norm_proto = proto[pred].norm(dim=(1, 2))
            norm_cps = cps.norm(dim=(1, 2))
            scores = (scores / (norm_proto * norm_cps)).detach().cpu().reshape(-1)
            self._record(ds_key, scores, score_name)

        return self._df