import abc
from pathlib import Path

import pandas as pd
import torch

class Score(metaclass=abc.ABCMeta):
    '''
    Base class for scores. Scores are stored in a long-format `pandas.DataFrame` with columns `['score name', 'loader', 'calib key', 'score value']`, one row per sample, saved at `self.path/self.name` with `torch.save()`. `compute()` skips the already computed `(score name, loader, calib key)` triplets, so calling a score again only computes the missing loaders.

    The `'calib key'` column holds the negative loader a value was calibrated against, for the scores whose values depend on it (e.g. `CAMExpScore`, `DMDScore`), and is `None` for all the others. It disambiguates the several values a same sample gets under a same `'score name'`.

    Inheriting classes must implement `compute()` and `_save_fitting()`/`load()`. Scores which have a `fit()` save the fitted values at `self._fit_file` at the end of `fit()`, the ones which do not fit anything implement them as no-ops.
    '''

    def __init__(self, **kwargs):
        '''
        Args:
        - path (str|pathlib.Path): folder where the scores `DataFrame` is saved.
        - name (str): name of the score. Used as the file name within `path`, and as the value of the `'score name'` column. Inheriting classes default it to their own name.
        '''
        self.path = Path(kwargs['path'])
        self.name = kwargs['name']

        # file with the saved DataFrame
        self._file = self.path/self.name

        # file with the values computed in fit(), for the scores which fit
        self._fit_file = Path(self._file.as_posix()+'.fitting')

        if self._file.exists():
            self._df = torch.load(self._file, weights_only=False)
        else:
            self._df = pd.DataFrame({
                'score name': pd.Series(dtype=str),
                'loader': pd.Series(dtype=str),
                'calib key': pd.Series(dtype=str),
                'score value': pd.Series(dtype=float)
                })
        return

    def _is_computed(self, **kwargs):
        '''
        Check whether a score was already computed for a given loader.

        Args:
        - ds_key (str): loader key.
        - calib_key (str): negative loader the score was calibrated against. Defaults to `None`.
        - name (str): score name. Defaults to `self.name`.
        '''
        ds_key = kwargs['ds_key']
        calib_key = kwargs.get('calib_key')
        name = kwargs.get('name', self.name)

        calib = self._df['calib key'].isna() if calib_key is None else self._df['calib key'] == calib_key

        return ((self._df['score name'] == name) & (self._df['loader'] == ds_key) & calib).any()

    def _record(self, **kwargs):
        '''
        Append the scores of a loader to `self._df` and save it to disk.

        Args:
        - ds_key (str): loader key.
        - scores (torch.Tensor|list[float]): one score per sample.
        - calib_key (str): negative loader the score was calibrated against. Defaults to `None`.
        - name (str): score name. Defaults to `self.name`.
        '''
        ds_key = kwargs['ds_key']
        scores = kwargs['scores']
        calib_key = kwargs.get('calib_key')
        name = kwargs.get('name', self.name)

        if hasattr(scores, 'tolist'):
            scores = scores.tolist()

        new_rows = pd.DataFrame({
            'score name': [name]*len(scores),
            'loader': [ds_key]*len(scores),
            'calib key': [calib_key]*len(scores),
            'score value': scores,
            })
        self._df = pd.concat([self._df, new_rows], ignore_index=True)

        self.path.mkdir(parents=True, exist_ok=True)
        torch.save(self._df, self._file)
        return

    @abc.abstractmethod
    def compute(self, **kwargs):
        '''
        Compute the score, skipping the loaders which are already in `self._df`, and return it.
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def _save_fitting(self, **kwargs):
        '''
        Save the values computed in `fit()` to `self._fit_file`. Called at the end of `fit()`.
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, **kwargs):
        '''
        Load the values computed in `fit()` from `self._fit_file`, returning 1 if the score is ready to be computed and 0 otherwise, so that it can be used as `if not score.load(): score.fit(...)`.
        '''
        raise NotImplementedError()
