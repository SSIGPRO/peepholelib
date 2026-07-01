from pathlib import Path
import pandas as pd
import torch

_COLS = ['dataset', 'score name', 'score value']

class Score:
    score_name = None  # override in subclass

    def __init__(self, save_path=None):
        self.save_path = Path(save_path) if save_path is not None else None
        if self.save_path is not None and self.save_path.exists():
            self._df = torch.load(self.save_path, weights_only=False)
        else:
            self._df = pd.DataFrame({'dataset': pd.Series(dtype=str),
                                     'score name': pd.Series(dtype=str),
                                     'score value': pd.Series(dtype=float)})

    def _is_computed(self, ds_key, score_name=None):
        sn = score_name if score_name is not None else self.score_name
        return ((self._df['dataset'] == ds_key) & (self._df['score name'] == sn)).any()

    def _get_pending_loaders(self, loaders, score_name=None):
        pending = [k for k in loaders if not self._is_computed(k, score_name)]
        if pending:
            sn = score_name if score_name is not None else self.score_name
            print(f'{sn}: {len(pending)}/{len(loaders)} loaders still to compute: {pending}')
        return pending

    def _record(self, ds_key, scores, score_name=None):
        sn = score_name if score_name is not None else self.score_name
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        new_rows = pd.DataFrame({
            'dataset': [ds_key] * len(scores),
            'score name': [sn] * len(scores),
            'score value': scores,
        })
        self._df = pd.concat([self._df, new_rows], ignore_index=True)
        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._df, self.save_path)

    def remove(self, dataset=None, score_name=None):
        if dataset is None and score_name is None:
            raise ValueError("Provide at least one of 'dataset' or 'score_name'")
        mask = pd.Series(True, index=self._df.index)
        if dataset is not None:
            mask &= self._df['dataset'] == dataset
        if score_name is not None:
            mask &= self._df['score name'] == score_name
        self._df = self._df[~mask].reset_index(drop=True)
        if self.save_path is not None:
            torch.save(self._df, self.save_path)

    @property
    def df(self):
        return self._df

    def __call__(self, **kwargs):
        loaders = kwargs.get('loaders')
        score_name = kwargs.get('score_name', self.score_name)
        if loaders is not None:
            pending = self._get_pending_loaders(loaders, score_name)
            if not pending:
                return self._df
            kwargs = {**kwargs, 'loaders': pending}
        return self._compute(**kwargs)

    def _compute(self, **kwargs):
        raise NotImplementedError