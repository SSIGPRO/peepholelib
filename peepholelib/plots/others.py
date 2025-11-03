# General pytho stuff
from pathlib import Path as Path
from math import ceil
import numpy as np

# plotting stuff
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import seaborn as sb
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, auc

# torch stuff
import torch
from torcheval.metrics import BinaryAUROC as AUC
from torchmetrics.classification import BinaryROC as ROC

def plot_attacks(**kwargs):
    '''
    Plot attacks and samples of the original distribution. The samples are selected only if the original image has been classified correctly by the reference model and the attack was succesful in fooling the model.

    Args:
    - attacks (list[str]): list of attacks that are analysed
    - score type (list[str]): list of scores deployed
    - scores (dict(str:dict(str: torch.tensor))): Three-level dictionary with first keys being the attacks, second-level the loader name, third-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'scores'. Defaults to 'None'.
    - path ('str'): Path to save plots.
    - verbose (bool): print progress messages.
    '''

    scores = kwargs.get('scores')
    score_type = kwargs.get('score_type')
    attacks = kwargs.get('attacks')
    loaders = kwargs.get('loaders', None)
    path = kwargs.get('path', None)
    

    # parse arguments
    if path == None: 
        path = Path.cwd()
    else:
        path = Path(path)
    
    fig_type, axs_type = plt.subplots(1,len(score_type), figsize=(16,8))
    fig_type.suptitle(f'AUC based on type')
    fig_atk, axs_atk = plt.subplots(1, len(attacks), figsize=(16,8))
    fig_atk.suptitle(f'AUC based on attack')
    for j ,type in enumerate(score_type):

        for k, attack in enumerate(attacks):
            
            fig, axs = plt.subplots(1,3, figsize=(16,8))
            fig.suptitle(f'Attack {attack}')

            for i, ds_key in enumerate(loaders):
                s = scores[attack][ds_key][type]
                if isinstance(s, torch.Tensor):
                    s = scores[attack][ds_key][type].detach().cpu().numpy()
                
                n_samples = len(s)//2
                axs[i].hist(s[:n_samples], bins=50, color='red', label=f'{attack}', alpha=0.5)
                axs[i].hist(s[n_samples:], bins=50, color='green',label='Original', alpha=0.5)
                y_true = np.zeros_like(s, dtype=int)
                y_true[n_samples:] = 1

                fpr, tpr, thresholds = roc_curve(y_true, s)

                # if ds_key == 'val':
                #     fpr_target = 0.05
                #     keep = np.where(fpr <= fpr_target)[0]
                #     idx = keep[-1]
                #     threshold_fpr = thresholds[idx]
                #     axs[i].axvline(threshold_fpr, color='black', linestyle='--', label=f"th={threshold_fpr:.2f}")
                #     axs[i].set_title(f'Val - FPR @ 5%')
                #     axs[i].legend()
                # elif ds_key == 'test':
                #     idx_test = np.argmin(np.abs(thresholds - threshold_fpr))

                #     fpr_at_threshold = fpr[idx_test]
                #     axs[i].axvline(threshold_fpr, color='black', linestyle='--', label=f"th={threshold_fpr:.2f}")
                #     axs[i].set_title(f'Test - FPR @ {fpr_at_threshold*100:.2f}%')
                # else:
                #     raise RuntimeError('Do not mess with other portion of the dataset just val and test')
                
                auc_value = roc_auc_score(y_true, s)

                axs[2].plot(fpr, tpr, lw=2, label=f"{ds_key} = {auc_value:.3f})")
                if ds_key == 'val':
                    axs[2].plot([0, 1], [0, 1], color='gray', linestyle='--', label="Chance (AUC = 0.5)")

                axs[2].set_xlim([0.0, 1.0])
                axs[2].set_ylim([0.0, 1.05])
                axs[2].set_xlabel("False Positive Rate")
                axs[2].set_ylabel("True Positive Rate")
                axs[2].set_title("Receiver Operating Characteristic")
                axs[2].legend(loc="lower right")
                axs[2].grid(True)

                if ds_key == 'test':
                    axs_type[j].plot(fpr, tpr, lw=2, label=f"{attack} = {auc_value:.3f}")
                    axs_type[j].set_xlim([0.0, 1.0])
                    axs_type[j].set_ylim([0.0, 1.05])
                    axs_type[j].set_xlabel("False Positive Rate")
                    axs_type[j].set_ylabel("True Positive Rate")
                    axs_type[j].set_title(f"{type}")
                    axs_type[j].legend(loc="lower right")
                    axs_type[j].grid(True)

                    if len(attacks)!=1:
                        _axs_atk = axs_atk[k]
                    else:
                        _axs_atk = axs_atk

                    _axs_atk.plot(fpr, tpr, lw=2, label=f"{type} = {auc_value:.3f}")
                    _axs_atk.set_xlim([0.0, 1.0])
                    _axs_atk.set_ylim([0.0, 1.05])
                    _axs_atk.set_xlabel("False Positive Rate")
                    _axs_atk.set_ylabel("True Positive Rate")
                    _axs_atk.set_title(f"{attack}")
                    _axs_atk.legend(loc="lower right")
                    _axs_atk.grid(True)
                
            fig.savefig((path/f'{attack}_{type}_distribution.png').as_posix(), dpi=300, bbox_inches='tight')    
    fig_type.savefig((path/f'AUC_based_type.png').as_posix(), dpi=300, bbox_inches='tight')     
    fig_atk.savefig((path/f'AUC_based_atk.png').as_posix(), dpi=300, bbox_inches='tight')      
    return
    
def plot_ROC_confidence(**kwargs):
    '''
    Plot ROC curve for indistribution case. Confidences are computed for a score threshold 'th', assuming values bellow 'th' are wrongly classified and above are correctly classified true positive and negative ('TP(th)' and 'TN(th)') are computed, so 'conf(th) = (TP(th)+TN(th))/ns'. 'th' is plotted from 0 to 'max_score'. 

    Args:
    - corevectors (peepholelib.coreVectors.CoreVectors): corevectors with dataset parsed (see `peepholelib.coreVectors.parse_ds`).
    - scores (dict(str:dict(str: torch.tensor))): Two-level dictionary with first keys being the loader name, seconde-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'scores'. Defaults to 'None'.
    - path ('str'): Path to save plots.
    - max_score (float): Max score for the accuracy plot, within '[0., 1.]'.
    - verbose (bool): print progress messages.
    '''

    cvs = kwargs.get('corevectors')
    scores = kwargs.get('scores')
    loaders = kwargs.get('loaders', None)
    path = kwargs.get('path', None)

    # parse arguments
    if path == None: 
        path = Path.cwd()
    else:
        path = Path(path)

    if loaders == None: loaders = list(scores.keys())

    # save AUCs for plotting 
    aucs_df = pd.DataFrame()
    drop_df = pd.DataFrame()

    for loader_n, ds_key in enumerate(loaders):
        fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(11, 5))

        for score_n, score_name in enumerate(scores[ds_key].keys()):
            _scores = scores[ds_key][score_name]
            results = cvs._dss[ds_key]['result']  
            if not isinstance(results, torch.Tensor):
                results = torch.tensor(results)
            y_true = results.to(torch.int64).cpu().numpy()

            # scores as floats
            y_score = _scores.detach().to(torch.float32).cpu().numpy()

            # If "lower is better", flip:
            # y_score = -y_score

            fpr, tpr, thr = roc_curve(y_true, y_score)   # sklearn handles thresholds

            idx = np.searchsorted(tpr, 0.95, side="left")  # first tpr >= 0.95
            if idx < len(fpr):
                fpr95 = float(fpr[idx])
                thr95 = float(thr[idx])   # score threshold achieving that point
            else:  # safety if TPR never reaches 0.95
                fpr95 = 1.0
                thr95 = thr[-1]

            line_main, = ax_main.plot(fpr, tpr, label=f"{score_name} (FPR@95%: {fpr95:.3f})")
            col = line_main.get_color()

            # use the same color for the dot
            ax_zoom.plot(fpr, tpr, color=col)
            ax_zoom.plot([fpr95], [0.95], marker="o", markersize=6, color=col)

        for ax in (ax_main, ax_zoom):
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)            
            ax.axhline(0.95, linestyle=":", linewidth=1.5, color="gray")    
            ax.grid(True, linestyle=":", linewidth=0.5)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
        ax_zoom.set_xlim(0.5, 0.8)
        ax_zoom.set_ylim(0.9, 1.0)

        ax_main.set_xlabel("False Positive Rate")
        ax_main.set_ylabel("True Positive Rate")
        ax_main.set_title(f"ROC — {ds_key} (full)")
        ax_zoom.set_xlabel("False Positive Rate")
        ax_zoom.set_title("Zoom: FPR ∈ [0.5, 0.8]")

        # Legend only on the main plot (applies to both since colors match)
        ax_main.legend(loc="lower right", fontsize=9)
        plt.tight_layout()

        plt.savefig((path/f'confidence_AUC.png').as_posix(), dpi=300, bbox_inches='tight')
        plt.close()
    return 

def FPR95_OOD_AA(**kwargs):
    '''
    Computes the FPR at 95% TPR for both OOD and Adversarial Attacks

    Args:
    - corevectors (peepholelib.coreVectors.CoreVectors): corevectors with dataset parsed (see `peepholelib.coreVectors.parse_ds`).
    - scores (dict(str:dict(str: torch.tensor))): Two-level dictionary with first keys being the loader name, seconde-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'scores'. Defaults to 'None'.
    - verbose (bool): print progress messages.
    '''

    cvs = kwargs.get('corevectors')
    scores = kwargs.get('scores')
    val_loader = kwargs.get('val_loader', None)
    test_loaders = kwargs.get('test_loaders', None)

    scores_name = list(scores[val_loader].keys())

    for sn in scores_name:
        s_val = scores[val_loader][sn]
        results_val = cvs._dss[val_loader]['result']

        s_oks = s_val[results_val == True]
        sorted_pos, _ = torch.sort(s_oks, descending=True)
        tpr95_index = int(torch.ceil(torch.tensor(0.95 * sorted_pos.numel())).item()) - 1
        threshold = sorted_pos[tpr95_index]

        for test in test_loaders:
            s_test = scores[test][sn]
            results_test = cvs._dss[test]['result']
            s_kos = s_test[results_test == False]
            fpr95 = (s_kos >= threshold).float().mean().item()
            print(f'FPR95 for {test} {sn} split: {fpr95:.4f}')

    return
