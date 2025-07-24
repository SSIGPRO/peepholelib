# torch stuff
import torch
from torch.nn.functional import softmax as sm

def conceptogram_protoclass_score(**kwargs):
    '''
    Compute the Proto-Class score of all conceptograms in `phs._phs[`loaders`]`. `target_modules` are passed to `ph.get_conceptograms()` so the evaluation only consider the indicated modules. The score is computed by comparing the conceptogram with the protoclasses. #TODO: Add paper or a full description.

    Args:
    - peepholes (peepholelib.peepholes.Peepholes): peepholes from which to take the conceptograms.
    - corevectors (peepholelib.coreVectors.CoreVectors): corevectors respective to the `phs`.
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'peepholes._phs'. Defaults to 'None'.
    - target_modules (list[str]): list if target modules, as keys from the model `statedict`. If 'None' uses all modules in 'peepholes._phs[loaders[0]]'.
    - proto_key (str): The key in `loaders` to get compute the protoclasses from.
    - proto_th (float): Model's confidence threshold to select samples for the protoclass computation ('0 <= proto_th <= 1').
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.

    Returns
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Proto-Class'. If 'append_scores' is passed, the dictionaries are appended.
    - proto: Protoclasses, an array of shape '(nc, nd, nc)', with 'nc' the number of classes and 'nd' the number of modules in 'target_modules'. Each element in the first dim is the protoclass of the respective label.
    '''

    phs = kwargs.get('peepholes')
    cvs = kwargs.get('corevectors')
    loaders = kwargs.get('loaders', None)
    target_modules = kwargs.get('target_modules', None)
    proto_key = kwargs.get('proto_key', 'train')
    proto_th = kwargs.get('proto_threshold', 0.9)
    append_scores = kwargs.get('append_scores', None)
    verbose = kwargs.get('verbose', False)
    
    # parse arguments
    score_name = 'Proto-Class'
    if loaders == None: loaders = list(phs._phs.keys())
    if target_modules == None: target_modules = list(phs._phs[loaders[0]].keys())

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
    # get conceptogram 
    cpss = phs.get_conceptograms(loaders=loaders, target_modules=target_modules, verbose=verbose)

    # sizes and values just to facilitate 
    nd = cpss[loaders[0]].shape[1] # number of layers (distributions)
    nc = cpss[loaders[0]].shape[2] # number of classes

    cps = cpss[proto_key]
    results = cvs._dss[proto_key]['result']
    labels = cvs._dss[proto_key]['label']
    confs = sm(cvs._dss[proto_key]['output'], dim=-1).max(dim=-1).values
    
    # compute proto-classes
    proto = torch.zeros(nc, nd, nc)
    for i in range(nc):
        cl = torch.logical_and(labels == i, results == 1)
        idx = torch.logical_and(cl, confs>proto_th)
        
        _p = cps[idx].sum(dim=0)  ## P'_j
        _p /= _p.sum(dim=1, keepdim=True)
        proto[i][:] = _p[:]

    # compute protoclass score
    for ds_key in loaders:
        cps = cpss[ds_key]
        results = cvs._dss[ds_key]['result']
        pred = (cvs._dss[ds_key]['pred']).int()

        scores = (proto[pred]*cps).sum(dim=(1,2))
        scores = scores/(proto[pred].norm(dim=(1,2))*cps.norm(dim=(1,2)))

        ret[ds_key][score_name] = scores 
        
    return ret, proto 

def model_confidence_score(**kwargs):
    '''
    Compute the model confidence score all samples in 'cvs._dss[`loaders`]'. The score is computed by comparing the conceptogram with the protoclasses. #TODO: Add paper or a full description.

    Args:
    - corevectors (peepholelib.coreVectors.CoreVectors): corevectors with dataset parsed (see `peepholelib.coreVectors.parse_ds`).
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'corevectors._dss'. Defaults to 'None'.
    - append_scores (dict): Append the scores in this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.

    Returns:
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Model-Confidence'. If 'append_scores' is passed, the dictionaries are appended.
    '''

    cvs = kwargs.get('corevectors')
    loaders = kwargs.get('loaders', None)
    append_scores = kwargs.get('append_scores', None)
    verbose = kwargs.get('verbose', False)
    
    # parse arguments
    score_name = 'Model-Confidence'
    if loaders == None: loaders = list(cvs._dss.keys())

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
    for loader_n, ds_key in enumerate(loaders):
        scores = sm(cvs._dss[ds_key]['output'], dim=-1).max(dim=-1).values
        ret[ds_key][score_name] = scores 
        
    return ret 
