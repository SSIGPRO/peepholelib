def has_score(**kwargs):
    '''
    Return `True` if `name` exists in `scores` for every base_loader-inference combination in `inference_names`.

    Args:
    - scores (pandas.DataFrame): scores `DataFrame` (see `peepholelib.scores.score.Score.df`).
    - name (str): score name to look for.
    - inference_names (dict{str: list[str]}): `{base_loader: [inference, ...]}`.
    '''
    scores = kwargs['scores']
    name = kwargs['name']
    inference_names = kwargs['inference_names']

    for base_loader, inferences in inference_names.items():
        for inf in inferences:
            loader_key = f'{base_loader}-{inf}'
            if not ((scores['dataset'] == loader_key) & (scores['score name'] == name)).any():
                return False
    return True


def pair_score_name(**kwargs):
    '''
    Name used by the scores computed for a pair of loaders (see `peepholelib.scores.dmd.DMDScore` and `peepholelib.scores.cam.CAMExpScore`), which also identifies the negative loader.

    Args:
    - name (str): score name.
    - loader (str): negative test loader.
    '''
    name = kwargs['name']
    loader = kwargs['loader']

    return f'{name}-{loader}'
