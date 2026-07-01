from peepholelib.scores.score import Score

class MaxLogitScore(Score):
    score_name = 'MaxLogit'

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
            scores = dss._dss[ds_key]['output'].max(dim=-1).values
            self._record(ds_key, scores.detach().cpu().reshape(-1))

        return self._df