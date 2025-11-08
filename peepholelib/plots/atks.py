# General pytho stuff
from pathlib import Path as Path
from math import ceil

# plotting stuff
from sklearn.metrics import roc_curve, roc_auc_score, auc

# torch stuff
import torch
from torcheval.metrics import BinaryAUROC as AUC

def auc_atks(**kwargs):
    '''
    Plot OOD detection.

    Args:
    - scores (dict(str:dict(str: torch.tensor))): Two-level dictionary with first keys being the loader name, seconde-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - id_loaders (dict(str:str|list(str))): Dictionary of loaders of in-distribution data, with the key being the score type and values a str or list of strings for respective loaders.
    - ood_loaders (list[str]): out-of-distribution loaders to consider

    - verbose (bool): print progress messages.
    '''
    scores = kwargs['scores']
    dss = kwargs['datasets']
    filter_key = kwargs.get('filter_key', 'attack_success')
    ori_loaders = kwargs.get('ori_loaders')
    atk_loaders = kwargs.get('atk_loaders')
    verbose = kwargs.get('verbose', False)

    for loader_n, ds_key in enumerate(atk_loaders):

        # save in-distribution and out-of-distribution scores for plotting
        for score_n, score_name in enumerate(ori_loaders.keys()):
            _ori_loader = ori_loaders[score_name]

            if type(_ori_loader) is list:
                s_ori = scores[_ori_loader[loader_n]][score_name]
            else:
                s_ori = scores[_ori_loader][score_name]

            s_atk = scores[ds_key][score_name]
            idx = dss._dss[ds_key][filter_key] == 1
            s_ori = s_ori[idx]
            s_atk = s_atk[idx]

            # computing AUC for each score type
            _labels = torch.hstack((torch.ones(s_ori.shape), torch.zeros(s_atk.shape)))
            _scores = torch.hstack((s_ori, s_atk))

            auc = AUC().update(_scores, _labels).compute().item()
            if verbose: print(f'AUC for {ds_key} {score_name} split: {auc:.4f}')

    return 
