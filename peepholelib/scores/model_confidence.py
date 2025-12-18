from torch.nn.functional import softmax as sm
from peepholelib.scores.score import Score


    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'corevectors._dss'. Defaults to 'None'.
    - append_scores (dict): Append the scores in this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.

    Returns:
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Model-Confidence'. If 'append_scores' is passed, the dictionaries are appended.
    '''

    dss = kwargs.get('datasets')
    loaders = kwargs.get('loaders', None)
    append_scores = kwargs.get('append_scores', None)
    score_name = kwargs.get('score_name', 'MSP')
    verbose = kwargs.get('verbose', False)
    
    # parse arguments
    if loaders == None: loaders = list(dss._dss.keys())

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
            scores = sm(dss._dss[ds_key]['output'], dim=-1).max(dim=-1).values
            self._record(ds_key, scores.detach().cpu().reshape(-1))

        return self._df
