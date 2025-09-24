def nll_score(**kwargs):
    '''
    Compute Negative Log-Likelihood (NLL) scores using a provided driller model.
    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): Parsed datasets.
    - loaders (list[str]): loaders like ['train', 'test', 'val']
    - signal_key (str): 'residual' (error) or 'latent_space'
    - driller (peepholelib.peepholes.drill_base.DrillBase): Driller model to compute NLL scores.
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function.
    - score_name (str): name of the score to be stored in the return dictionary.
    '''
    dss = kwargs.get('datasets')
    loaders = kwargs.get('loaders', ['train', 'test', 'val'])
    signal_key = kwargs.get('signal_key', 'residual')  
    driller = kwargs.get('driller')
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


    driller.load()
    with driller as drl:
        drl.score_samples()
        for ds_key in loaders:
            if ds_key not in ret:
                ret[ds_key] = {}
            
            dss_flat = dss._dss[ds_key][signal_key].squeeze().flatten(start_dim=1)
            # get nll scores
            ret[ds_key][score_name] = drl.score_samples(dss_flat)
    
    return ret