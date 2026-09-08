# general python stuff
import math
from sklearn.metrics import roc_curve

# torch stuff
import torch

def CAM_lin_score(**kwargs):
    '''
    Compute the CAM linear score of all samples in `phs._phs[`loaders`]`. The score is `1 - eta`, with `eta` the cost of the model's predicted class averaged over `target_modules`, so that higher values indicate samples better covered by the trusted signature. Assumes the costs lie in [0, 1] (`normalize=True` in the driller).

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): Parsed datasets corresponding to `peepholes`. Used to retrieve the model's predicted class (`'pred'` key) for each sample.
    - peepholes (peepholelib.peepholes.peepholes.Peepholes): Peepholes containing the MRC cost `eta` for each loader and layer.
    - loaders (list[str]): Loaders to consider, if 'None', gets all loaders in 'peepholes._phs'. Defaults to 'None'.
    - target_modules (list[str]): Layers whose `eta` values are averaged. Defaults to all modules in `peepholes` for the first loader.
    - append_scores (dict): Append the scores in this dictionary to the scores computed in this function. Overwrite if same keys.
    - score_name (str): Key under which scores are stored for each loader. Defaults to `'CAM-lin'`.

    Returns:
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionary with the first key being the loaders, and second being the score name. If 'append_scores' is passed, the dictionaries are appended.
    '''
    dss = kwargs['datasets']
    phs = kwargs['peepholes']
    loaders = kwargs.get('loaders', None)
    target_modules = kwargs.get('target_modules', None)
    append_scores = kwargs.get('append_scores', None)
    score_name = kwargs.get('score_name', 'CAM-lin')

    # parse arguments
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
    for ds_key in loaders:
        h = sum(phs._phs[ds_key][layer] for layer in target_modules)/len(target_modules)
        pred = dss._dss[ds_key][:]['pred']
        ret[ds_key][score_name] = 1 - h.gather(1, pred.unsqueeze(1)).squeeze(1)

    return ret

def CAM_exp_score(**kwargs):
    '''
    Compute the CAM confidence score `c` for safe and unsafe samples. For each entry in `unsafe_loaders`, `tau` is calibrated per class using `safe_loader_train` and all corresponding unsafe train loaders via a AUC. Safe-test scores are stored under the first unsafe-train key. For each class, calibration unsafe samples are drawn equally from all unsafe train loaders so that the total unsafe count matches the safe count.

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): Parsed datasets corresponding to `peepholes`. Used to retrieve the model's predicted class (`'pred'` key) for each sample.
    - peepholes (peepholelib.peepholes.peepholes.Peepholes): Peepholes containing the MRC lambda confidence `eta` for each loader and layer.
    - safe_loader_train (str): Loader with trusted/safe samples used to calibrate `tau` per class. Defaults to `'val'`.
    - safe_loader_test (str): Loader with trusted/safe samples to score.
    - unsafe_loaders (dict[str, list[str]]): Maps each unsafe test loader to a list of unsafe train loaders used to calibrate `tau`.
    - target_modules (list[str]): Layers whose `eta` values are summed to form `h`. Defaults to all modules in `peepholes` for `safe_loader_train`.
    - append_scores (dict): Existing scores dict to extend. New scores are added (or overwritten) in-place on a shallow copy.
    - score_name (str): Key under which scores are stored for each loader. Defaults to `'CAM'`.

    Returns:
    - ret (dict[str, dict[str, torch.Tensor]]): Two-level dict keyed by loader name then score name, each value a 1-D tensor of per-sample scores in [0, 1]. Safe-test scores are stored under the key of the first unsafe train loader (same convention as `DMD_score()`).
    '''
    dss = kwargs['datasets']
    phs = kwargs['peepholes']
    safe_loader_train = kwargs.get('safe_loader_train', 'val')
    safe_loader_test = kwargs['safe_loader_test']
    unsafe_loaders = kwargs['unsafe_loaders']
    target_modules = kwargs.get('target_modules', None)
    append_scores = kwargs.get('append_scores', None)
    score_name = kwargs.get('score_name', 'CAM')

    if target_modules is None: target_modules = list(phs._phs[safe_loader_train].keys())

    if append_scores is not None:
        ret = dict(append_scores)
    else:
        ret = {}

    for unsafe_test_key, unsafe_train_loaders in unsafe_loaders.items():
        if unsafe_test_key not in ret:
            ret[unsafe_test_key] = dict()
        if unsafe_train_loaders[0] not in ret:
            ret[unsafe_train_loaders[0]] = dict()

    # accumulate h for the safe loaders once, outside the loop over unsafe pairs
    h_safe_train = sum(phs._phs[safe_loader_train][layer] for layer in target_modules)
    pred_safe_train = dss._dss[safe_loader_train][:]['pred']
    nl_model = h_safe_train.shape[1]

    h_safe_test = sum(phs._phs[safe_loader_test][layer] for layer in target_modules)
    pred_safe_test = dss._dss[safe_loader_test][:]['pred']
    h_pred_safe_test = h_safe_test.gather(1, pred_safe_test.unsqueeze(1)).squeeze(1)

    for unsafe_test_key, unsafe_train_loaders in unsafe_loaders.items():
        n_train = len(unsafe_train_loaders)

        h_unsafe_list = [
                sum(phs._phs[ul][layer] for layer in target_modules)
                for ul in unsafe_train_loaders
                ]
        pred_unsafe_list = [
                dss._dss[ul][:]['pred']
                for ul in unsafe_train_loaders
                ]

        # calibrate tau: Youden's J on the ROC for each class
        tau = torch.zeros(nl_model, device=h_safe_train.device)
        for i in range(nl_model):
            hi_safe = h_safe_train[pred_safe_train == i, i]
            n_safe_i = hi_safe.shape[0]

            if n_safe_i == 0:
                tau[i] = float('nan')
                continue

            # draw n_safe_i // n_train samples from each unsafe train loader
            n_per_loader = n_safe_i // n_train
            hi_unsafe_parts = []
            skip_class = False
            for h_unsafe, pred_unsafe, ul in zip(h_unsafe_list, pred_unsafe_list, unsafe_train_loaders):
                candidates = h_unsafe[pred_unsafe == i, i]
                if candidates.shape[0] == 0:
                    tau[i] = float('nan')
                    skip_class = True
                    break
                n_take = min(n_per_loader, candidates.shape[0])
                idx = torch.randperm(candidates.shape[0], device=candidates.device)[:n_take]
                hi_unsafe_parts.append(candidates[idx])
            if skip_class:
                continue

            hi_unsafe = torch.cat(hi_unsafe_parts)

            y_true = torch.cat((torch.zeros_like(hi_safe), torch.ones_like(hi_unsafe)))
            y_score = torch.cat((hi_safe, hi_unsafe))
            fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_score.cpu().numpy())
            tau[i] = float(thresholds[(tpr - fpr).argmax()])

        nan_mask = tau.isnan()
        if nan_mask.any():
            tau[nan_mask] = tau[~nan_mask].mean()

        # score safe test samples
        ret[unsafe_train_loaders[0]][score_name] = torch.exp(-h_pred_safe_test * math.log(2) / tau[pred_safe_test])

        # score unsafe test samples
        h_unsafe_test = sum(phs._phs[unsafe_test_key][layer] for layer in target_modules)
        pred_unsafe_test = dss._dss[unsafe_test_key][:]['pred']
        h_pred_unsafe_test = h_unsafe_test.gather(1, pred_unsafe_test.unsqueeze(1)).squeeze(1)
        ret[unsafe_test_key][score_name] = torch.exp(-h_pred_unsafe_test * math.log(2) / tau[pred_unsafe_test])

    return ret
