# torch stuff
import torch
from torch.nn.functional import softmax as sm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm

def RelU_score(**kwargs):
    '''
    Compute the Relative Uncertainty score as described in https://arxiv.org/abs/2306.01710.

    Args:
    - corevectors (peepholelib.coreVectors.CoreVectors): corevectors respective to the `phs`.
    - loaders (list[str]): loaders to consider, usually `['train', 'test', 'val']`, if `None`, gets all loaders in `corevectors._dss`. Defaults to `None`.
    - fit_key (str): loader to fit distributions. Usually `'train'`. Defaults to `'train'`.
    - lbd (float): lambda factor. Defaults to 0.5.
    - temperature (float): temperature factor. Defaults to 1.0.
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.

    '''

    cvs = kwargs.get('corevectors')
    loaders = kwargs.get('loaders', None)
    fit_key = kwargs.get('fit_key', 'train')
    lbd = kwargs.get('lbd', 0.5)
    temperature = kwargs.get('temperature', 1.)
    append_scores = kwargs.get('append_scores', None)
    verbose = kwargs.get('verbose', False)

    # parse arguments
    if loaders == None: loaders = list(cvs._dss.keys())
    score_name = 'Rel-U'

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
    results = cvs._dss[fit_key]['result']
    outputs = sm(cvs._dss[fit_key]['output']/temperature, dim=-1)

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
        outputs = sm(cvs._dss[ds_key]['output']/temperature, dim=-1)
        _params = torch.tril(params, diagonal=-1)
        _params = _params + _params.T
        _params = _params / _params.norm()
        scores = torch.diag(outputs@_params@outputs.T) 
        ret[ds_key][score_name] = 1-((scores - s_min)/(s_max - s_min)).clip(0., 1.)

    return ret

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
    - train_data
    - train_label (numpy.array): samples used to train the Logistic regresser.
    - test_data
    - test_label (numpy.array) samples used to validate the trained Logistic regeressor.

    Returns
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Proto-Class'. If 'append_scores' is passed, the dictionaries are appended.
    '''
    # TODO: fix docs

    train_data = kwargs.get('train_data')
    train_label = kwargs.get('train_label')
    test_data = kwargs.get('test_data')

    # You can use torch tensors for the LogisticRegressionCV, no need to numpy
    lr = LogisticRegressionCV(n_jobs=-1,max_iter=5000).fit(train_data, train_label)

    y_train = lr.predict_proba(train_data)[:, 1]
    y_test = lr.predict_proba(test_data)[:, 1]

    return y_train, y_test 

def ood_aware_DMD_score(**kwargs):
    '''
    Compute the DMD score by training a linear regressor on the a validation portion of the ood dataset 

    Args:
    - peepavg (peepholelib.peepholes.Peepholes): peepholes from which we compute the linear regressor.
    - coreavg (peepholelib.coreVectors.CoreVectors): corevectors respective to the `phs`.
    - ood-loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'peepholes._phs'. Defaults to 'None'.
    - target_modules (list[str]): list if target modules, as keys from the model `statedict`. If 'None' uses all modules in 'peepholes._phs[loaders[0]]'.
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.

    Returns
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Proto-Class'. If 'append_scores' is passed, the dictionaries are appended.
    - proto: Protoclasses, an array of shape '(nc, nd, nc)', with 'nc' the number of classes and 'nd' the number of modules in 'target_modules'. Each element in the first dim is the protoclass of the respective label.
    '''
    # TODO: fix documentation

    phs = kwargs.get('peepholes')
    id_loader_train = kwargs.get('id_loader_train', 'val')
    id_loader_test = kwargs.get('id_loader_test', 'test')
    ood_loaders_train = kwargs.get('ood_loaders_train', None)
    ood_loaders_test = kwargs.get('ood_loaders_test', None)
    target_modules = kwargs.get('target_modules', None)
    append_scores = kwargs.get('append_scores', None)

    # parse arguments
    score_name = 'DMD-Aware'
    if target_modules == None: target_modules = list(phs._phs[id_loader_train].keys())

    # create the return dictionary. 
    if append_scores != None:
        ret = dict(append_scores)
    else: ret = {}

    for ds_key in ood_loaders_train:
        if not ds_key in ret:
            ret[ds_key] = dict()

    for ds_key in ood_loaders_test:
        if not ds_key in ret:
            ret[ds_key] = dict()


    #-----------
    # computations
    #-----------

    # it would be better to stack fisrt then get the max
    train_ori = torch.stack([phs._phs[id_loader_train][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
    test_ori = torch.stack([phs._phs[id_loader_test][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
    for ood_train_key, ood_test_key in zip(ood_loaders_train, ood_loaders_test):
        train_ood = torch.stack([phs._phs[ood_train_key][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
        train_data = torch.vstack((train_ori, train_ood))
    
        train_label = torch.hstack((torch.ones(len(train_ori)), torch.zeros(len(train_ood))))

        test_ood = torch.stack([phs._phs[ood_test_key][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
        test_data = torch.vstack((test_ori, test_ood))

        _, y_test = DMD_score(
                train_data=train_data,
                train_label=train_label,
                test_data=test_data,
                )

        ret[ood_train_key][score_name] = torch.tensor(y_test)[:len(test_ori)]
        ret[ood_test_key][score_name] = torch.tensor(y_test)[len(test_ori):]

    return ret

# better namek
def DMD_aware_atk(**kwargs):
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
    
    label_ori = np.ones(len(train_ori))
    label_atk = np.zeros(len(train_atk))
    train_label = np.concatenate((label_ori, label_atk), axis=0)

    idx = torch.argwhere((cvs._dss['test']['result']==1) & (cvs_atk._dss['test']['attack_success']==1)).squeeze()

    test_ori = torch.stack([phs._phs['test'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)[idx].detach().cpu().numpy()
    test_atk = torch.stack([phs_atk._phs['test'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)[idx].detach().cpu().numpy()
    
    test_data = np.concatenate((test_ori, test_atk), axis=0)
    label_ori = np.ones(len(test_ori))
    label_atk = np.zeros(len(test_atk))
    test_label = np.concatenate((label_ori, label_atk), axis=0)

    ret_ = DMD_score(train_data=train_data, train_label=train_label,
                     test_data=test_data, test_label=test_label)

    ret['val'][score_name] = ret_['val']
    ret['test'][score_name] = ret_['test']

    return ret

# give a better name
def DMD_unaware_atk(**kwargs):
    '''
    Compute the DMD score by training a linear regressor on the original dataset and the attack dataset. We consider as training and test samples attacks crafted from different attack algorithm 

    Args:
    - peepavg (peepholelib.peepholes.Peepholes): peepholes from which we compute the linear regressor.
    - coreavg (peepholelib.coreVectors.CoreVectors): corevectors respective to the `phs`.
    - coreavg_atk_train (dict): dictionary containing as keys the attacks used as training of the regressor and as values the corresponding coreavg
    - peepavg_atk_train (dict): dictionary containing as keys the attacks used as training of the regressor and as values the corresponding peepavg
    - coreavg_atk_test (peepholelib.peepholes.CoreVectors): coreavg of the attack under test
    - peepavg_atk_test (peepholelib.coreVectors.Peepholes): peepavg of the attack under test
    - target_modules (list[str]): list if target modules, as keys from the model `statedict`. If 'None' uses all modules in 'peepholes._phs[loaders[0]]'.
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.

    Returns
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Proto-Class'. If 'append_scores' is passed, the dictionaries are appended.
    '''

    coreavg_atk_train = kwargs.get('coreavg_atk_train')
    peepavg_atk_train = kwargs.get('peepavg_atk_train')
    coreavg_atk_test = kwargs.get('coreavg_atk_test')
    peepavg_atk_test = kwargs.get('peepavg_atk_test')
    ph = kwargs.get('peepavg')
    cv = kwargs.get('coreavg')  
    loaders = kwargs.get('loaders', None) 
    target_modules = kwargs.get('target_modules', None)
    append_scores = kwargs.get('append_scores', None)
    atk_name = kwargs.get('atk_name')

    # parse arguments
    n_samples = 3333
    score_name = 'DMD-Unaware'
    
    if loaders == None: loaders = list(ph._phs.keys())
    if target_modules == None: target_modules = list(ph._phs[loaders[0]].keys())

    # create the return dictionary. 
    if append_scores != None:
        ret = dict(append_scores)
    else:
        ret = {}
    for ds_key in loaders:
        if not ds_key in ret:
            ret[ds_key] = dict()

    #------------------
    # computations
    #------------------

    #------------------
    #  TRAINING
    #------------------
                
    f_ori = torch.stack([ph._phs['val'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1).detach().cpu().numpy()
    f_atk = {}
                
    for i, atk_name in enumerate(peepavg_atk_train.keys()):
            
            f_atk[atk_name] = torch.stack([peepavg_atk_train[atk_name]._phs['val'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)
            idx = torch.argwhere((cv._dss['val']['result']==1) & (coreavg_atk_train[atk_name]._dss['val']['attack_success']==1)).squeeze()
            
            assert len(idx) >= n_samples, f'Not enough samples for attack {atk_name} in training set. Found {len(idx)} samples, expected at least {n_samples}.'

            rand = torch.randperm(len(idx))[:n_samples]
            f_atk[atk_name] = f_atk[atk_name][idx[rand]]
   
    f_atk = torch.concat([f_atk[atk_name] for atk_name in peepavg_atk_train.keys()], dim=0)#.detach().cpu().numpy()   
    
    label_ori = np.ones(len(f_ori))
    label_atk = np.zeros(len(f_atk))
    
    train_data = np.concatenate((f_ori, f_atk), axis=0)
    train_label = np.concatenate((label_ori, label_atk), axis=0)  

    #------------------
    #  VALIDATION
    #------------------ 

    idx = torch.argwhere((cv._dss['val']['result']==1) & (coreavg_atk_test._dss['val']['attack_success']==1)).squeeze()
    f_ori = torch.stack([ph._phs['val'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)[idx].detach().cpu().numpy()
    f_atk = torch.stack([peepavg_atk_test._phs['val'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)[idx].detach().cpu().numpy()

    label_ori = np.ones(len(f_ori))
    label_atk = np.zeros(len(f_atk))

    val_data = np.concatenate((f_ori, f_atk), axis=0)
    val_label = np.concatenate((label_ori, label_atk), axis=0) 

    ret_ = DMD_score(train_data=train_data, train_label=train_label,
                    test_data=val_data, test_label=val_label)
    
    ret['val'][score_name] = ret_['test']

    #------------------
    #  TEST
    #------------------

    idx = torch.argwhere((cv._dss['test']['result']==1) & (coreavg_atk_test._dss['test']['attack_success']==1)).squeeze()
    f_ori = torch.stack([ph._phs['test'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)[idx].detach().cpu().numpy()
    f_atk = torch.stack([peepavg_atk_test._phs['test'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)[idx].detach().cpu().numpy()

    label_ori = np.ones(len(f_ori))
    label_atk = np.zeros(len(f_atk))

    test_data = np.concatenate((f_ori, f_atk), axis=0)
    test_label = np.concatenate((label_ori, label_atk), axis=0)
    ret_ = DMD_score(train_data=train_data, train_label=train_label,
                    test_data=test_data, test_label=test_label)

    ret['test'][score_name] = ret_['test']

    return ret

# give a better name
def FeatureSqueezing(**kwargs):

    '''
    Compute the Feature Squeezing score 

    Args:
    - corevectors (peepholelib.coreVectors.CoreVectors): corevectors respective to the `phs`.
    - corevectors_atk(peepholelib.peepholes.CoreVectors): corevectors of the attack under test
    - Detector (peepholelib.featureSqueezing.FeatureSqueezingDetector): Detector used to analyse the input images
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.
    - devie 

    Returns
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Proto-Class'. If 'append_scores' is passed, the dictionaries are appended.
    '''

    cva = kwargs.get('corevectors_atk')
    cv = kwargs.get('corevectors') 
    device = kwargs.get('device') 
    detector = kwargs.get('detector', None) 
    loaders = kwargs.get('loaders', None)
    append_scores = kwargs.get('append_scores', None)
    score_name = kwargs.get('score_name', 'Feature-Squeezing')
    bs = 64
    
    
    if loaders == None: loaders = list(cv._dss.keys())
    if append_scores != None:
        ret = dict(append_scores)
    else:
        ret = {}
    for ds_key in loaders:
        if not ds_key in ret:
            ret[ds_key] = dict()

    loaders_ori = {ds_key: DataLoader(cv._dss[ds_key]['image'], batch_size=bs, shuffle=False, num_workers=4) for ds_key in loaders}
    loaders_atk = {ds_key: DataLoader(cva._dss[ds_key]['image'], batch_size=bs, shuffle=False, num_workers=4) for ds_key in loaders}
    for (ds_key, dlo), (_, dla) in zip(loaders_ori.items(), loaders_atk.items()):
        print(f'------{ds_key}------')
        print(f'Computing score for Original Dataset')

        idx = torch.argwhere((cv._dss[ds_key]['result']==1) & (cva._dss[ds_key]['attack_success']==1))
        ori_list = []

        for inputs in tqdm(dlo):

            inputs = inputs.to(device)

            output = detector(inputs)

            ori_list.append(output.detach().cpu())

        s_ori = torch.cat(ori_list, dim=0)
        s_ori = s_ori[idx]

        print(f'Computing score for Attack Dataset')

        atk_list = []

        for inputs in tqdm(dla):

            inputs = inputs.to(device)

            output = detector(inputs)

            atk_list.append(output.detach().cpu())

        s_atk = torch.cat(atk_list, dim=0)
        s_atk = s_atk[idx]

        scores = torch.cat([s_atk, s_ori], dim=0)
        ret[ds_key][score_name] = scores

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
