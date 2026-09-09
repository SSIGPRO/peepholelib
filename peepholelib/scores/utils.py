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
