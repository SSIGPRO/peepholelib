# general python stuff
from math import floor

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
    Compute the DMD score by training a linear regressor on two portions of the datasets. It considers on sample as positive samples, and many as negative ones, training one regressor for each negative loader.
    Since the score of the positive samples change for each negative loader used in training, the scores of the positive samples are saved with the keys of the negative loaders used for training. It is confusing, and we need a beeter way to structure these scores.

    Args:
    - peepholes (peepholelib.peepholes.Peepholes): peepholes from which we compute the linear regressor.
    - pos_loader_train (str): loader to consider as positive samples for training. Typically in-distribution for OOD or original samples for attacks.
    - pos_loader_test (str): loader to consider as positive samples for testing.
    - neg_loaders (dict{str: list[str]}): dictionary with keys for negative samples. The key correspond to the TEST loader, and the value is a list of loaders used as negative samples for training.
    - target_modules (list[str]): list if target modules, as keys from the model `state_dict`. If 'None' uses all modules in 'peepholes._phs[pos_loader_train]'.
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.

    Returns
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Proto-Class'. If 'append_scores' is passed, the dictionaries are appended.
    '''

    phs = kwargs.get('peepholes')
    pos_loader_train = kwargs.get('pos_loader_train', 'val')
    pos_loader_test = kwargs.get('pos_loader_test', 'test')
    neg_loaders = kwargs.get('neg_loaders')
    target_modules = kwargs.get('target_modules', None)
    append_scores = kwargs.get('append_scores', None)
    score_name = kwargs.get('score_name', 'DMD')

    # parse arguments
    if target_modules == None: target_modules = list(phs._phs[pos_loader_train].keys())

    # create the return dictionary. 
    if append_scores != None:
        ret = dict(append_scores)
    else: ret = {}

    for ds_key in neg_loaders.keys():
        if not ds_key in ret:
            ret[ds_key] = dict()

        if not neg_loaders[ds_key][0] in ret:
            ret[neg_loaders[ds_key][0]] = dict()

    #-----------
    # computations
    #-----------

    # it would be better to stack fisrt then get the max
    train_pos = torch.stack([phs._phs[pos_loader_train][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
    test_pos = torch.stack([phs._phs[pos_loader_test][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
    
    nps = len(train_pos) # number of positive samples
    idx = torch.randperm(nps)

    for neg_test_key, neg_train_loaders in neg_loaders.items():
        nnl = len(neg_train_loaders) # number of negative loaders
        nspnl = floor(nps/nnl) # number sampler per neg loader
        
        # get nspnl samples for each negative loader
        train_neg = []
        for i, nl in enumerate(neg_train_loaders):
            _train_neg = torch.stack([phs._phs[nl][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
            train_neg.append(_train_neg[idx[i*nspnl:(i+1)*nspnl]])
        train_neg = torch.vstack(train_neg)

        # train data and labels
        train_data = torch.vstack((train_pos, train_neg))
        train_label = torch.hstack((torch.ones(len(train_pos)), torch.zeros(len(train_neg))))
        # test data
        test_neg = torch.stack([phs._phs[neg_test_key][layer]['peepholes'].max(dim=1)[0] for layer in target_modules], dim=1)
        test_data = torch.vstack((test_pos, test_neg))

        _, y_test = __DMD_score__(
                train_data = train_data,
                train_label = train_label,
                test_data = test_data,
                )

        ret[neg_train_loaders[0]][score_name] = torch.tensor(y_test)[:len(test_pos)]
        ret[neg_test_key][score_name] = torch.tensor(y_test)[len(test_pos):]
    return ret
