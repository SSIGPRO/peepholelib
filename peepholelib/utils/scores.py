# torch stuff
import torch
from torch.nn.functional import softmax as sm
from sklearn.linear_model import LogisticRegressionCV
import numpy as np

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
    - score_name (string): score name if specification in needed (adv attacks case)
    - proto proto: Protoclasses, an array of shape '(nc, nd, nc)', with 'nc' the number of classes and 'nd' the number of modules in 'target_modules'. Each element in the first dim is the protoclass of the respective label.

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
    proto = kwargs.get('proto', None)
    
    # parse arguments
    
    if loaders == None: loaders = list(phs._phs.keys())
    if target_modules == None: target_modules = list(phs._phs[loaders[0]].keys())
    score_name = 'Proto-Class'

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

    if proto == None:
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

def DMD_score(**kwargs):
    '''
    Compute the DMD score by training a linear regressor on the original dataset and the attack dataset. We consider as training and test samples attacks crafted with the same algorithm 

    Args:
    - train_data, train_label (numpy arrays): samples used to train the Logistic regresser.
    - test_data, test_label (numpy.arrays) samples used to validate the trained Logistic regeressor.

    Returns
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Proto-Class'. If 'append_scores' is passed, the dictionaries are appended.
    '''

    train_data = kwargs.get('train_data')
    train_label = kwargs.get('train_label')

    test_data = kwargs.get('test_data')
    test_label = kwargs.get('test_label')

    lr = LogisticRegressionCV(n_jobs=-1,max_iter=5000).fit(train_data, train_label)

    y_val = lr.predict_proba(train_data)[:, 1]
    y_test = lr.predict_proba(test_data)[:, 1]

    ret = dict()

    ret['val'] = y_val
    ret['test'] = y_test

    return ret


def DMD_aware(**kwargs):
    '''
    Compute the DMD score by training a linear regressor on the original dataset and the attack dataset. We consider as training and test samples attacks crafted with the same algorithm 

    Args:
    - peepavg (peepholelib.peepholes.Peepholes): peepholes from which we compute the linear regressor.
    - coreavg (peepholelib.coreVectors.CoreVectors): corevectors respective to the `phs`.
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'peepholes._phs'. Defaults to 'None'.
    - target_modules (list[str]): list if target modules, as keys from the model `statedict`. If 'None' uses all modules in 'peepholes._phs[loaders[0]]'.
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.

    Returns
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Proto-Class'. If 'append_scores' is passed, the dictionaries are appended.
    - proto: Protoclasses, an array of shape '(nc, nd, nc)', with 'nc' the number of classes and 'nd' the number of modules in 'target_modules'. Each element in the first dim is the protoclass of the respective label.
    '''

    phs = kwargs.get('peepavg')
    cvs = kwargs.get('coreavg')
    phs_atk = kwargs.get('peepavg_atk')
    cvs_atk = kwargs.get('coreavg_atk')
    loaders = kwargs.get('loaders', None)
    target_modules = kwargs.get('target_modules', None)
    append_scores = kwargs.get('append_scores', None)
    verbose = kwargs.get('verbose', False)

    # parse arguments
    score_name = 'DMD-Aware'
    if loaders == None: loaders = list(phs_atk._phs.keys())
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

    idx = torch.argwhere((cvs._dss['val']['result']==1) & (cvs_atk._dss['val']['attack_success']==1)).squeeze()

    train_ori = torch.stack([phs._phs['val'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)[idx].detach().cpu().numpy()
    train_atk = torch.stack([phs_atk._phs['val'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)[idx].detach().cpu().numpy()
    
    train_data = np.concatenate((train_ori, train_atk), axis=0)
    
    label_ori = np.zeros(len(train_ori))
    label_atk = np.ones(len(train_atk))
    train_label = np.concatenate((label_ori, label_atk), axis=0)

    idx = torch.argwhere((cvs._dss['test']['result']==1) & (cvs_atk._dss['test']['attack_success']==1)).squeeze()

    test_ori = torch.stack([phs._phs['test'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)[idx].detach().cpu().numpy()
    test_atk = torch.stack([phs_atk._phs['test'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)[idx].detach().cpu().numpy()
    
    test_data = np.concatenate((test_ori, test_atk), axis=0)
    label_ori = np.zeros(len(test_ori))
    label_atk = np.ones(len(test_atk))
    test_label = np.concatenate((label_ori, label_atk), axis=0)

    ret_ = DMD_score(train_data=train_data, train_label=train_label,
                     test_data=test_data, test_label=test_label)

    ret['val'][score_name] = ret_['val']
    ret['test'][score_name] = ret_['test']

    return ret

def DMD_unaware(**kwargs):
    '''
    Compute the DMD score by training a linear regressor on the original dataset and the attack dataset. We consider as training and test samples attacks crafted from different attack algorithm 

    Args:
    - peepavg (peepholelib.peepholes.Peepholes): peepholes from which we compute the linear regressor.
    - coreavg (peepholelib.coreVectors.CoreVectors): corevectors respective to the `phs`.
    - peepavg_atk_dict (dict): The dictionary defines the training and the test set
                               of the Linear regressor. It is composed of a training and a test key
                               within train we have the attacks used for training the regressor and each value
                               corresponds to peepholelib.peepholes.Peepholes while test is the single 
                               test attack on which we evaluate the performances of the regressor
    - coreavg_atk_dict (dict): The dictionary defines the training and the test set
                               of the Linear regressor. It is composed of a training and a test key
                               within train we have the attacks used for training the regressor and each value
                               corresponds to peepholelib.coreVectors.CoreVectors while test is the single 
                               test attack on which we evaluate the performances of the regressor
    
    - target_modules (list[str]): list if target modules, as keys from the model `statedict`. If 'None' uses all modules in 'peepholes._phs[loaders[0]]'.
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.

    Returns
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Proto-Class'. If 'append_scores' is passed, the dictionaries are appended.
    '''

    peepavg_atk_dict = kwargs.get('peepavg_atk_dict')
    coreavg_atk_dict = kwargs.get('coreavg_atk_dict')
    ph = kwargs.get('peepavg')
    cv = kwargs.get('coreavg')
    
    target_modules = kwargs.get('target_modules', None)
    append_scores = kwargs.get('append_scores', None)
    verbose = kwargs.get('verbose', False)
    atk_name = kwargs.get('atk_name')

    # parse arguments
    n_samples = 3333
    score_name = 'DMD-Unaware'
    
    if loaders == None: loaders = list(peepavg_atk_dict[0].value._phs.keys())
    if target_modules == None: target_modules = list(ph._phs[loaders[0]].keys())

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

    for config in peepavg_atk_dict:
        ph_atk = peepavg_atk_dict[config]['train']
        ph_atk_test = peepavg_atk_dict[config]['test']

        cv_atk = coreavg_atk_dict[config]['train']
        cv_atk_test = coreavg_atk_dict[config]['test']

                
        f_ori = torch.stack([ph._phs['val'][layer]['score_max'] for layer in target_modules],dim=1).detach().cpu().numpy()
        f_atk = []
        
        for i, pha, cva in enumerate(zip(ph_atk, cv_atk)):
                
            f_atk[i] = torch.stack([pha._phs['val'][layer]['score_max'] for layer in target_modules],dim=1)
            idx = torch.argwhere((cv._dss['val']['result']==1) & (cva._dss['val']['attack_success']==1)).squeeze()
            
            assert len(idx) >= n_samples, f'Not enough samples for attack in training set. Found {len(idx)} samples, expected at least {n_samples}.'

            rand = torch.randperm(len(idx))[:n_samples]
            f_atk[i] = f_atk[i][idx[rand]]
                            
            f_atk = torch.concat(f_atk, dim=0).detach().cpu().numpy()

            label_ori = np.zeros(len(f_ori))
            label_atk = np.ones(len(f_atk))
            train_data = np.concatenate((f_ori, f_atk), axis=0)
            train_label = np.concatenate((label_ori, label_atk), axis=0)

                
        idx = torch.argwhere((cv._dss['test']['result']==1) & (cv_atk_test._dss['test']['attack_success']==1)).squeeze()
        f_ori = torch.stack([ph._phs['test'][layer]['score_max'] for layer in target_modules],dim=1)[idx].detach().cpu().numpy()
        f_atk = torch.stack([ph_atk_test._phs['test'][layer]['score_max'] for layer in target_modules],dim=1)[idx].detach().cpu().numpy()

        label_ori = np.zeros(len(f_ori))
        label_atk = np.ones(len(f_atk))

        test_data = np.concatenate((f_ori, f_atk), axis=0)
        test_label = np.concatenate((label_ori, label_atk), axis=0)

        ret_ = DMD_score(train_data, train_label,
                     test_data, test_label)

        ret[atk_name]['val'][score_name] = ret_['val']
        ret[atk_name]['test'][score_name] = ret_['test']

        ret[atk_name]['val']['label'] = train_label
        ret[atk_name]['test']['label'] = test_label

    return ret

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
