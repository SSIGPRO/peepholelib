# torch stuff
import torch
from torch.nn.functional import softmax as sm

def RelU_score(**kwargs):
    '''
    Compute the Relative Uncertainty score as described in https://arxiv.org/abs/2306.01710.

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - loaders (list[str]): loaders to consider, usually `['train', 'test', 'val']`, if `None`, gets all loaders in `datasets._dss`. Defaults to `None`.
    - fit_key (str): loader to fit distributions. Usually `'train'`. Defaults to `'train'`.
    - lbd (float): lambda factor. Defaults to 0.5.
    - temperature (float): temperature factor. Defaults to 1.0.
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.

    '''

    dss = kwargs.get('datasets')
    loaders = kwargs.get('loaders', None)
    fit_key = kwargs.get('fit_key', 'train')
    lbd = kwargs.get('lbd', 0.5)
    temperature = kwargs.get('temperature', 1.)
    append_scores = kwargs.get('append_scores', None)
    verbose = kwargs.get('verbose', False)
    score_name = kwargs.get('score_name', 'Rel-U')

    # parse arguments
    if loaders == None: loaders = list(dss._dss.keys())
    

    # create the return dictionary. 
    if append_scores != None:
        ret = dict(append_scores)
    else:
        ret = {}
    
    for ds_key in loaders:
        if not ds_key in ret:
            ret[ds_key] = dict()
    
    #-----------
    # computations
    #-----------
    results = dss._dss[fit_key]['result']
    outputs = sm(dss._dss[fit_key]['output']/temperature, dim=-1)

    train_probs_pos = outputs[results == 1] 
    train_probs_neg = outputs[results == 0] 

    d_pos = torch.einsum("ij,ik->ijk", train_probs_pos, train_probs_pos).mean(dim=0)
    d_neg = torch.einsum("ij,ik->ijk", train_probs_neg, train_probs_neg).mean(dim=0)
    params = -(1 - lbd)*d_pos + lbd*d_neg 
    params = torch.tril(params, diagonal=-1)
    params = params + params.T
    params = torch.relu(params)
    params = params / params.norm()

    _scores = torch.diag(outputs@params@outputs.T) 
    s_min = _scores.min()
    s_max = _scores.max()

    for ds_key in loaders:
        outputs = sm(dss._dss[ds_key]['output']/temperature, dim=-1)
        _params = torch.tril(params, diagonal=-1)
        _params = _params + _params.T
        _params = _params / _params.norm()
        scores = torch.diag(outputs@_params@outputs.T) 
        ret[ds_key][score_name] = 1-((scores - s_min)/(s_max - s_min)).clip(0., 1.)

    return ret
