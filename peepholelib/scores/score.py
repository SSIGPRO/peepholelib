import abc
from pathlib import Path

import pandas as pd
import torch


class Score(metaclass=abc.ABCMeta):
    '''
    Base class for scores. Scores are stored in a long-format `pandas.DataFrame` with columns `['dataset', 'score name', 'score value']`, one row per sample, saved at `self.path/self.name` with `torch.save()`. Already computed `(dataset, score name)` pairs are skipped, so calling a score again only computes the missing loaders.

    Inheriting classes must implement `_compute()`, called by `__call__()`, and `_save_fitting()`/`load()`. Scores which have a `fit()` save the fitted values at `self._fit_file` at the end of `fit()`, the ones which do not fit anything implement them as no-ops.
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
                'dataset': pd.Series(dtype=str),
                'score name': pd.Series(dtype=str),
                'score value': pd.Series(dtype=float)
                })
        return

    def _is_computed(self, **kwargs):
        '''
        Check whether a score was already computed for a given loader.

        Args:
        - ds_key (str): loader key.
        - name (str): score name. Defaults to `self.name`.
        '''
        ds_key = kwargs['ds_key']
        name = kwargs.get('name', self.name)

        return ((self._df['dataset'] == ds_key) & (self._df['score name'] == name)).any()

    def _get_pending_loaders(self, **kwargs):
        '''
        Return the loaders which are still to compute.

        Args:
        - loaders (list[str]): loaders to consider.
        - name (str): score name. Defaults to `self.name`.
        '''
        loaders = kwargs['loaders']
        name = kwargs.get('name', self.name)

        pending = [k for k in loaders if not self._is_computed(ds_key=k, name=name)]
        if pending:
            print(f'{name}: {len(pending)}/{len(loaders)} loaders still to compute: {pending}')
        return pending

    def _record(self, **kwargs):
        '''
        Append the scores of a loader to `self._df` and save it to disk.

        Args:
        - ds_key (str): loader key.
        - scores (torch.Tensor|list[float]): one score per sample.
        - name (str): score name. Defaults to `self.name`.
        '''
        ds_key = kwargs['ds_key']
        scores = kwargs['scores']
        name = kwargs.get('name', self.name)

        if hasattr(scores, 'tolist'):
            scores = scores.tolist()

        new_rows = pd.DataFrame({
            'dataset': [ds_key]*len(scores),
            'score name': [name]*len(scores),
            'score value': scores,
            })
        self._df = pd.concat([self._df, new_rows], ignore_index=True)

        self.path.mkdir(parents=True, exist_ok=True)
        torch.save(self._df, self._file)
        return

    def remove(self, **kwargs):
        '''
        Remove rows from `self._df` and save it to disk. At least one of `dataset` or `name` must be given.

        Args:
        - dataset (str): remove the rows of this loader.
        - name (str): remove the rows of this score name.
        '''
        dataset = kwargs.get('dataset', None)
        name = kwargs.get('name', None)

        if dataset is None and name is None:
            raise ValueError("Provide at least one of 'dataset' or 'name'")

        mask = pd.Series(True, index=self._df.index)
        if dataset is not None:
            mask &= self._df['dataset'] == dataset
        if name is not None:
            mask &= self._df['score name'] == name

        self._df = self._df[~mask].reset_index(drop=True)

        self.path.mkdir(parents=True, exist_ok=True)
        torch.save(self._df, self._file)
        return

    @property
    def df(self):
        return self._df

    def __call__(self, **kwargs):
        '''
        Compute the score, skipping the loaders which were already computed. All `kwargs` are forwarded to `self._compute()`.

        Args:
        - loaders (list[str]): loaders to consider. If given, it is filtered down to the pending ones before calling `self._compute()`. Scores which do not take a `loaders` argument (e.g. the ones taking pairs of loaders) do their own filtering within `_compute()`.
        '''
        loaders = kwargs.get('loaders', None)

        if loaders is not None:
            pending = self._get_pending_loaders(loaders=loaders)
            if len(pending) == 0:
                return self._df
            kwargs = {**kwargs, 'loaders': pending}

        return self._compute(**kwargs)

    @abc.abstractmethod
    def _compute(self, **kwargs):
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
