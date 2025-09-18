# torch stuff
import torch
from torch.nn.functional import softmax as sm

def model_confidence_score(**kwargs):
    '''
    Compute the model confidence score all samples in 'cvs._dss[`loaders`]'. The score is computed as the max(softmax(model output's)).

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
    verbose = kwargs.get('verbose', False)
    
    # parse arguments
    score_name = 'MSP'
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
    for loader_n, ds_key in enumerate(loaders):
        scores = sm(dss._dss[ds_key]['output'], dim=-1).max(dim=-1).values
        ret[ds_key][score_name] = scores 
        
    return ret 
