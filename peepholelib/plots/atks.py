# General pytho stuff
from pathlib import Path as Path
from math import ceil

# plotting stuff
from sklearn.metrics import roc_curve, roc_auc_score, auc

# torch stuff
import torch
from torcheval.metrics import BinaryAUROC as AUC

def auc_fpr(**kwargs):
    '''
    Compute and print OOD or AA AUC and `1 - FPR` at 95% TPR scores. Both metrics are computed over the same samples. The FPR is computed with the threshold taken at 95% TPR over the in-distribution scores, and is the fraction of out-of-distribution (or attacked) samples scored above it.

    Args:
    - scores (dict(str:dict(str: torch.tensor))): Two-level dictionary with first keys being the loader name, seconde-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - datasets (peepholelib.datasets.parsedDatataset.ParsedDataset))): parsed datasets. 
    - filter_key (str): String within `datasets._dss[loader]` to filters the classification results, e.g. if you want to select only AAs which were successfull. If `None`, all samples are used. Defaults to `'attack_sucess'`
    - ori_loaders (dict(str:str|list(str))): Dictionary of loaders of in-distribution data, with the key being the score type and values a str or list of strings for respective loaders.
    - ood_loaders (list[str]): out-of-distribution loaders to consider

    - verbose (bool): print progress messages.

    Returns:
    - aucs (dict(str:dict(str: float))): AUCs with first keys being the loader name and second-level key the score names.
    - fprs (dict(str:dict(str: float))): `1 - FPR` at 95% TPR, with the same keys as `aucs`.
    '''
    scores = kwargs['scores']
    dss = kwargs['datasets']
    filter_key = kwargs.get('filter_key', 'attack_success')
    ori_loaders = kwargs.get('ori_loaders')
    atk_loaders = kwargs.get('atk_loaders')
    verbose = kwargs.get('verbose', False)

    aucs = {}
    fprs = {}
    for loader_n, ds_key in enumerate(atk_loaders):
        aucs[ds_key] = {}
        fprs[ds_key] = {}
        
        # save in-distribution and out-of-distribution scores for plotting
        for score_n, score_name in enumerate(ori_loaders.keys()):
            _ori_loader = ori_loaders[score_name]

            if type(_ori_loader) is list:
                s_ori = scores[_ori_loader[loader_n]][score_name]
            else:
                s_ori = scores[_ori_loader][score_name]

            s_atk = scores[ds_key][score_name]

            if filter_key is not None:
                idx = dss._dss[ds_key][filter_key] == 1
                s_ori = s_ori[idx]
                s_atk = s_atk[idx]
            else:
                # guarantees the same number of samples
                _ns = min(len(s_ori), len(s_atk))
                s_ori = s_ori[torch.randperm(len(s_ori))[:_ns]]
                s_atk = s_atk[torch.randperm(len(s_atk))[:_ns]]

            # computing AUC for each score type
            _labels = torch.hstack((torch.ones(s_ori.shape), torch.zeros(s_atk.shape)))
            _scores = torch.hstack((s_ori, s_atk))

            auc = AUC().update(_scores, _labels).compute().item()

            # computing 1-FPR@95 for each score type, using the threshold
            # keeping 95% of the in-distribution samples
            _sorted, _ = torch.sort(s_ori, descending=True)
            _th = _sorted[ceil(0.95*_sorted.numel())-1]
            fpr95 = 1 - (s_atk >= _th).float().mean().item()

            if verbose: print(f'AUC for {ds_key} {score_name} split: {auc:.4f}, 1-FPR@95: {fpr95:.4f}')
            aucs[ds_key][score_name] = auc
            fprs[ds_key][score_name] = fpr95

    return aucs, fprs
