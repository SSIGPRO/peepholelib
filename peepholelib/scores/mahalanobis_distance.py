import torch

def mahalanobis_distance_score(**kwargs):  
    '''
    Compute Mahalanobis distance scores based on the residuals/latent space representations. Mahalanobis distance can be an aproximation or not based on 'use sigma'
    Args:
    - datasets (Sentinel)
    - loaders (list[str]): loaders like ['train', 'test'. 'val']
    - fit_key (str): loader to fit distributions. Usually 'train' 
    - signal_key (str): 'residual' (error) or 'latent_space' (encripted version)
    - use_sigma (bool): whether to use covariance matrix or identity matrix (aproximation)
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - score_name (str): name of the score to be stored in the return dictionary.
    
    '''
    corevectors = kwargs['corevectors']
    loaders = kwargs.get('loaders', ['train', 'test', 'val'])
    fit_key = kwargs.get('fit_key', 'train')
    signal_key = kwargs.get('signal_key', 'residual')  
    use_sigma = kwargs.get('use_sigma', False)
    append_scores = kwargs.get('append_scores', None)
    score_name = kwargs.get('score_name', 'mahalanobis')

    # create the return dictionary. 
    if append_scores != None:
        ret = dict(append_scores)
    else:
        ret = {}
    
    for ds_key in corevectors._corevds.keys():
        if not ds_key in ret:
            ret[ds_key] = dict()

    #--------------
    # Computations
    #--------------
    fit_signal = corevectors._corevds[fit_key][signal_key]
    fit_signal = fit_signal.squeeze().flatten(start_dim=1)
    mean = fit_signal.mean(axis=0)
    cv = mean.shape[0]
    
    if use_sigma: # covariance matrix
        sigma = torch.cov(fit_signal.T)        
        sigma_inv = torch.linalg.pinv(sigma)    

    # Mahalanobis distances
    for ds_key in corevectors._corevds.keys():
        test_signal = corevectors._corevds[ds_key][signal_key].squeeze().flatten(start_dim=1)
        diff = (test_signal - mean) 
        
        # if sigma is identity matrix then Mahalanobis is just the norm
        if use_sigma:
            _d = diff.unsqueeze(1)
            dists = (_d@sigma_inv@(_d.transpose(1,2))).squeeze().sqrt()
        else:
            dists = diff.norm(dim=1)

        ret[ds_key][score_name] = dists
    return ret


def mahalanobis_distance_score_peepholes(**kwargs):  
    '''
    Compute Mahalanobis distance scores based on the residuals/latent space representations. Mahalanobis distance can be an aproximation or not based on 'use sigma'
    Args:
    - datasets (Sentinel)
    - loaders (list[str]): loaders like ['train', 'test'. 'val']
    - fit_key (str): loader to fit distributions. Usually 'train' 
    - signal_key (str): 'residual' (error) or 'latent_space' (encripted version)
    - use_sigma (bool): whether to use covariance matrix or identity matrix (aproximation)
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - score_name (str): name of the score to be stored in the return dictionary.
    
    '''
    peepholes = kwargs['peepholes']
    loaders = kwargs.get('loaders', ['train', 'test', 'val'])
    fit_key = kwargs.get('fit_key', 'train')
    signal_key = kwargs.get('signal_key', 'residual')  
    use_sigma = kwargs.get('use_sigma', False)
    append_scores = kwargs.get('append_scores', None)
    score_name = kwargs.get('score_name', 'mahalanobis')

    # create the return dictionary. 
    if append_scores != None:
        ret = dict(append_scores)
    else:
        ret = {}
    
    for ds_key in peepholes._phs.keys():
        if not ds_key in ret:
            ret[ds_key] = dict()

    #--------------
    # Computations
    #--------------
    fit_signal = peepholes._phs[fit_key][signal_key]['peepholes']
    fit_signal = fit_signal.squeeze().flatten(start_dim=1)
    mean = fit_signal.mean(axis=0)
    cv = mean.shape[0]
    
    if use_sigma: # covariance matrix
        sigma = torch.cov(fit_signal.T)        
        sigma_inv = torch.linalg.pinv(sigma)    

    # Mahalanobis distances
    for ds_key in peepholes._phs.keys():
        test_signal = peepholes._phs[ds_key][signal_key]['peepholes'].squeeze().flatten(start_dim=1)
        diff = (test_signal - mean) 
        
        # if sigma is identity matrix then Mahalanobis is just the norm
        if use_sigma:
            _d = diff.unsqueeze(1)
            dists = (_d@sigma_inv@(_d.transpose(1,2))).squeeze().sqrt()
        else:
            dists = diff.norm(dim=1)

        ret[ds_key][score_name] = dists
    return ret
   



