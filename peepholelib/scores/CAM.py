# general python stuff
import math
from sklearn.metrics import roc_curve

# torch stuff
import torch

def CAM_score(**kwargs):
    '''
    Compute the CAM confidence score `c` for safe and unsafe samples. For each entry in `unsafe_loaders`, `tau` is calibrated per class using `safe_loader_train` and all corresponding unsafe train loaders via a ROC-based Youden's J criterion (Section C.2 of the supplementary material of Rossolini et al., IEEE TSE 2023), then scores are computed for `safe_loader_test` and the unsafe test loader as `c = exp(-h*ln(2)/tau_y_hat)` in [0, 1]. Safe-test scores are stored under the first unsafe-train key (same convention as `DMD_score()`). `h` for `safe_loader_train` and `safe_loader_test` is accumulated once, outside the loop over unsafe pairs. For each class, calibration unsafe samples are drawn equally from all unsafe train loaders so that the total unsafe count matches the safe count.

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
                raise RuntimeError(f'No samples predicted as class {i} in loader "{safe_loader_train}" to calibrate its threshold.')

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

        # score safe test samples (stored under first unsafe train key, as in DMD_score)
        ret[unsafe_train_loaders[0]][score_name] = torch.exp(-h_pred_safe_test * math.log(2) / tau[pred_safe_test])

        # score unsafe test samples
        h_unsafe_test = sum(phs._phs[unsafe_test_key][layer] for layer in target_modules)
        pred_unsafe_test = dss._dss[unsafe_test_key][:]['pred']
        h_pred_unsafe_test = h_unsafe_test.gather(1, pred_unsafe_test.unsqueeze(1)).squeeze(1)
        ret[unsafe_test_key][score_name] = torch.exp(-h_pred_unsafe_test * math.log(2) / tau[pred_unsafe_test])

    return ret
