def nll_score(**kwargs):
    '''
    Compute Negative Log-Likelihood (NLL) scores using a provided driller model.
    Args:
    - loaders (list[str]): loaders like ['train', 'test', 'val']
    - driller: already loaded driller
    - corevectors: already loaded corevectors
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function.
    - score_name (str): name of the score to be stored in the return dictionary.
    '''
    loaders = kwargs.get('loaders', ['train', 'test', 'val'])
    driller = kwargs.get('driller')
    _cvs = kwargs.get('corevectors')
    append_scores = kwargs.get('append_scores', None)
    score_name = kwargs.get('score_name', 'nll')

    # create the return dictionary. 
    if append_scores != None:
        ret = dict(append_scores)
    else:
        ret = {}
    
    for ds_key in loaders:
        if not ds_key in ret:
            ret[ds_key] = dict()

    for ds_key in loaders:
            if ds_key not in ret:
                ret[ds_key] = {}

            cvs = _cvs._corevds[ds_key]
            data = driller.parser(cvs=cvs)

            # get nll scores
            ret[ds_key][score_name] = driller._classifier.score_samples(data)
    
    return ret