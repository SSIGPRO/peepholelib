# torch stuff
import torch
from sklearn.linear_model import LogisticRegressionCV
import numpy as np

def DMD_base(**kwargs):
    '''
    Compute the DMD score based on the pre-logits activation(input activations of the last layer). In this case no training is needed and no backpropagation to compute the score
    - coreavg (peepholelib.coreVectors.CoreVectors): corevectors respective to the `phs`.
    - layer (str): string indicating the layer used for the score computation
    - id
    - driller (peepholelib.peepholes.DeepMahalnobisDistance.DMD): istance of the classifier used to compute the score
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.

    Returns
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Proto-Class'. If 'append_scores' is passed, the dictionaries are appended.
    '''

    id_loader = kwargs.get('id_loader', 'test')
    ood_loaders = kwargs.get('ood_loaders')
    device = kwargs.get('device')
    layer = kwargs.get('layer')
    cvs = kwargs.get('coreavg')
    driller = kwargs.get('driller')
    append_scores = kwargs.get('append_scores', None)

    score_name = 'dmd_base'

    data_ori = cvs._corevds[id_loader][layer].to(device)
    num_classes = driller.nl_model
    num_samples = data_ori.shape[0]

    # create the return dictionary. 
    if append_scores != None:
        ret = dict(append_scores)
    else: ret = {}

    if not id_loader in ret: ret[id_loader] = dict()

    for ds_key in ood_loaders:
        if not ds_key in ret:
            ret[ds_key] = dict()

    # computation
    class_scores = torch.zeros((num_samples, num_classes))
    for c in range(num_classes):
        tensor = data_ori - driller._means[c].view(1, -1)
        class_scores[:, c] = -torch.matmul(
            torch.matmul(tensor, driller._precision), tensor.t()).diag()

    ret[id_loader][score_name] = torch.max(class_scores, dim=1)[0]

    for ood in ood_loaders:
        data_ood = cvs._corevds[ood][layer].to(device)
        data_ori = cvs._corevds[id_loader][layer]

        class_scores = torch.zeros((num_samples, num_classes))
        for c in range(num_classes):
            tensor = data_ood - driller._means[c].view(1, -1)
            class_scores[:, c] = -torch.matmul(torch.matmul(tensor, driller._precision), tensor.t()).diag()

        ret[ood][score_name] = torch.max(class_scores, dim=1)[0]
    
    return ret

def DMD_plus(**kwargs):
    '''
    Compute the DMD score based on the pre-logits activation(input activations of the last layer). In this case no training is needed and no backpropagation to compute the score
    - coreavg (peepholelib.coreVectors.CoreVectors): corevectors respective to the `phs`.
    - layer (str): string indicating the layer used for the score computation
    - id
    - driller (peepholelib.peepholes.DeepMahalnobisDistance.DMD): istance of the classifier used to compute the score
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.

    Returns
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Proto-Class'. If 'append_scores' is passed, the dictionaries are appended.
    '''

    id_loader = kwargs.get('id_loader', 'test')
    ood_loaders = kwargs.get('ood_loaders')
    device = kwargs.get('device')
    layer = kwargs.get('layer')
    cvs = kwargs.get('coreavg')
    driller = kwargs.get('driller')
    append_scores = kwargs.get('append_scores', None)

    score_name = 'dmd_plus'

    data_ori = cvs._corevds[id_loader][layer].to(device)  
    
    data_ori /= torch.linalg.vector_norm(data_ori, ord=2, dim=1, keepdim=True) 
    
    num_classes = driller.nl_model
    num_samples = data_ori.shape[0]

    # create the return dictionary. 
    if append_scores != None:
        ret = dict(append_scores)
    else: ret = {}

    if not id_loader in ret: ret[id_loader] = dict()

    for ds_key in ood_loaders:
        if not ds_key in ret:
            ret[ds_key] = dict()

    # computation

    class_scores = torch.zeros((num_samples, num_classes))
    for c in range(num_classes):
        tensor = data_ori - driller._means[c].view(1, -1)
        class_scores[:, c] = -torch.matmul(
            torch.matmul(tensor, driller._precision), tensor.t()).diag()

    ret[id_loader][score_name] = torch.max(class_scores, dim=1)[0]

    for ood in ood_loaders:
        data_ood = cvs._corevds[ood][layer].to(device)
        data_ood /= torch.linalg.vector_norm(data_ood, ord=2, dim=1, keepdim=True) 

        class_scores = torch.zeros((num_samples, num_classes))
        for c in range(num_classes):
            tensor = data_ood - driller._means[c].view(1, -1)
            class_scores[:, c] = -torch.matmul(torch.matmul(tensor, driller._precision), tensor.t()).diag()

        ret[ood][score_name] = torch.max(class_scores, dim=1)[0]
    
    return ret

def __DMD_score__(**kwargs):
    '''
    Compute the DMD score by training a linear regressor on two datasets.

    Args:
    - train_data (torch.Tensor): train samples.
    - train_label (torch.Tensor): train labels.
    - test_data (torch.tensor): test samples.

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

def DMD_score(**kwargs):
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
    train_loaders = kwargs.get('train_loaders')
    test_loaders = kwargs.get('test_loaders')
    target_modules = kwargs.get('target_modules', None)
    append_scores = kwargs.get('append_scores', None)
    score_name = kwargs.get('score_name', 'DMD')
    verbose = kwargs.get('verbose', False)

    # parse arguments
    if target_modules == None: target_modules = list(phs._phs[list(train_loaders.keys())[0]].keys())

    # create the return dictionary. 
    if append_scores != None:
        ret = dict(append_scores)
    else: ret = {}

    for ds_key_train, ds_key_test in zip(train_loaders.keys(), test_loaders.keys()):
        if not ds_key_train in ret:
            ret[ds_key_train] = dict()
        if not ds_key_test in ret:
            ret[ds_key_test] = dict()

    #-----------
    # computations
    #----------- 
    for pos_key_train in train_loaders.keys()):
        # positive samples are In-distribution or non-attacked ones
        train_pos = torch.stack([phs._phs[pos_key_train][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
        
        # get negative samples from multiple loaders
        n_pos_samples = len(train_pos)
        n_neg_loaders = len(train_loaders[pos_key_train])
        n_spnl = floor(n_pos_samples/n_neg_loaders) # number sampler per neg loader
        for neg_key_train in train_loaders[pos_key_train]:
            train_neg = torch.stack([phs._phs[neg_key_train][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)

        train_data = torch.vstack((train_pos, train_neg))
        train_label = torch.hstack((torch.ones(len(train_pos)), torch.zeros(len(train_neg))))

    '''
    for pos_key_train, pos_key_test in zip(train_loaders.keys(), test_loaders.keys()):
        test_pos = torch.stack([phs._phs[pos_key_test][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
        test_ood = torch.stack([phs._phs[ood_test_key][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
        test_data = torch.vstack((test_ori, test_ood))

        _, y_test = __DMD_score__(
                train_data=train_data,
                train_label=train_label,
                test_data=test_data,
                )

        ret[ood_train_key][score_name] = torch.tensor(y_test)[:len(test_ori)]
        ret[ood_test_key][score_name] = torch.tensor(y_test)[len(test_ori):]
    '''
    return ret

def DMD_score_bak(**kwargs):
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
    score_name = kwargs.get('score_name', 'DMD-Aware')

    # parse arguments
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

        _, y_test = __DMD_score__(
                train_data=train_data,
                train_label=train_label,
                test_data=test_data,
                )

        ret[ood_train_key][score_name] = torch.tensor(y_test)[:len(test_ori)]
        ret[ood_test_key][score_name] = torch.tensor(y_test)[len(test_ori):]

    return ret

# better name
def DMD_aware_atk(**kwargs):
    '''
    Compute the DMD score by training a linear regressor on the original dataset and the attack dataset. We consider as training and test samples attacks crafted with the same algorithm 

    Args:
    - peepavg_ori (peepholelib.peepholes.peepholes.Peepholes): peepholes from which we compute the linear regressor.
    - peepavg_atk (peepholelib.peepholes.peepholes.Peepholes): peepholes from which we compute the linear regressor.
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'peepholes._phs'. Defaults to 'None'.
    - target_modules (list[str]): list if target modules, as keys from the model `statedict`. If 'None' uses all modules in 'peepholes._phs[loaders[0]]'.
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.

    Returns
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Proto-Class'. If 'append_scores' is passed, the dictionaries are appended.
    '''

    phs_ori = kwargs.get('peepavg_ori')
    phs_atk = kwargs.get('peepavg_atk')
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
    # TODO: this should be done in the evaluation functions
    #idx = torch.argwhere((cvs._dss['val']['result']==1) & (cvs_atk._dss['val']['attack_success']==1)).squeeze()

    # TODO: CLEAN THOSE Numpy!!!!!!!!!!!
    train_ori = torch.stack([phs_ori._phs['val'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
    train_atk = torch.stack([phs_atk._phs['val'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
    train_data = torch.concatenate((train_ori, train_atk), axis=0)
    
    label_ori = torch.ones(len(train_ori))
    label_atk = torch.zeros(len(train_atk))
    train_label = np.concatenate((label_ori, label_atk), axis=0)

    # TODO: this should be done in the evaluation functions
    #idx = torch.argwhere((cvs._dss['test']['result']==1) & (cvs_atk._dss['test']['attack_success']==1)).squeeze()

    test_ori = torch.stack([phs._phs['test'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
    test_atk = torch.stack([phs_atk._phs['test'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
    test_data = torch.concatenate((test_ori, test_atk), axis=0)
    
    y_train, y_test = __DMD_score__(
            train_data = train_data,
            train_label = train_label,
            test_data = test_data,
            test_label = test_label
            )
    
    # TODO: wtf keys?
    ret['val'][score_name] = y_train
    ret['test'][score_name] = y_test

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
            #idx = torch.argwhere((cv._dss['val']['result']==1) & (coreavg_atk_train[atk_name]._dss['val']['attack_success']==1)).squeeze()
            
            #assert len(idx) >= n_samples, f'Not enough samples for attack {atk_name} in training set. Found {len(idx)} samples, expected at least {n_samples}.'

            rand = torch.randperm(len(idx))[:n_samples]
            f_atk[atk_name] = f_atk[atk_name][idx[rand]]
   
    f_atk = torch.concat([f_atk[atk_name] for atk_name in peepavg_atk_train.keys()], dim=0)#.detach().cpu().numpy()   
    train_data = torch.concatenate((f_ori, f_atk), axis=0)
    
    label_ori = torch.ones(len(f_ori))
    label_atk = torch.zeros(len(f_atk))
    train_label = torch.concatenate((label_ori, label_atk), axis=0)  

    #------------------
    #  VALIDATION
    #------------------ 

    #idx = torch.argwhere((cv._dss['val']['result']==1) & (coreavg_atk_test._dss['val']['attack_success']==1)).squeeze()
    f_ori = torch.stack([ph._phs['val'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)
    f_atk = torch.stack([peepavg_atk_test._phs['val'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)
    val_data = torch.concatenate((f_ori, f_atk), axis=0)

    _, y_test = __DMD_score__(train_data=train_data, train_label=train_label, test_data=val_data)
    
    ret['val'][score_name] = y_test

    #------------------
    #  TEST
    #------------------
    #idx = torch.argwhere((cv._dss['test']['result']==1) & (coreavg_atk_test._dss['test']['attack_success']==1)).squeeze()
    f_ori = torch.stack([ph._phs['test'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)
    f_atk = torch.stack([peepavg_atk_test._phs['test'][layer]['peepholes'].max(dim=1)[0] for layer in target_modules],dim=1)

    label_ori = torch.ones(len(f_ori))
    label_atk = torch.zeros(len(f_atk))

    test_data = torch.concatenate((f_ori, f_atk), axis=0)
    test_label = torch.concatenate((label_ori, label_atk), axis=0)
    _, y_test = __DMD_score__(train_data=train_data, train_label=train_label, test_data=test_data)

    ret['test'][score_name] = y_test

    return ret
