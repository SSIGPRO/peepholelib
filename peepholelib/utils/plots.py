# General pytho stuff
from pathlib import Path as Path
from math import ceil
import numpy as np

# plotting stuff
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sb
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

# torch stuff
import torch
from torcheval.metrics import BinaryAUROC as AUC


def plot_ood(**kwargs):
    '''
    Plot OOD detection.

    Args:
    - scores (dict(str:dict(str: torch.tensor))): Two-level dictionary with first keys being the loader name, seconde-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - id_loaders (dict(str:str|list(str))): Dictionary of loaders of in-distribution data, with the key being the score type and values a str or list of strings for respective loaders.
    - ood_loaders (list[str]): out-of-distribution loaders to consider

    - path ('str'): Path to save plots.
    - suffix ('str'): Suffix to append to the plot's file name.
    - verbose (bool): print progress messages.
    '''
    scores = kwargs.get('scores')
    id_loaders = kwargs.get('id_loaders')
    ood_loaders = kwargs.get('ood_loaders')
    path = kwargs.get('path', None)
    suffix = kwargs.get('suffix', '')
    verbose = kwargs.get('verbose', False)

    # parse arguments
    if path == None: 
        path = Path.cwd()
    else:
        path = Path(path)

    fig, axs = plt.subplots(1, len(ood_loaders)+1, sharex='none', sharey='none', figsize=(5*(len(ood_loaders)+1), 5))

    colors = ['xkcd:cobalt', 'xkcd:bluish green', 'xkcd:light orange', 'xkcd:dark hot pink', 'xkcd:purplish']
    lines = ['--', '-']

    # save aucs for plotting 
    aucs_df = pd.DataFrame()

    for loader_n, ds_key in enumerate(ood_loaders):

        # save in-distribution and out-of-distribution scores for plotting
        df_idood = pd.DataFrame()
        cs_idood, ls_idood = {}, {} 
        for score_n, score_name in enumerate(id_loaders.keys()):
            _id_loader = id_loaders[score_name]
            
            if type(_id_loader) is list:
                s_id = scores[_id_loader[loader_n]][score_name] # TODO: rearragen iteration order
            else:
                s_id = scores[_id_loader][score_name] # TODO: rearragen iteration order

            s_ood = scores[ds_key][score_name]

            # computing AUC for each score type
            _labels = torch.hstack((torch.ones(s_id.shape), torch.zeros(s_ood.shape)))
            _scores = torch.hstack((s_id, s_ood))

            auc = AUC().update(_scores, _labels).compute().item()
            if verbose: print(f'AUC for {ds_key} {score_name} split: {auc:.4f}')
            aucs_df = aucs_df._append(
                    pd.DataFrame({
                        'AUC': [auc],
                        'score name': [score_name],
                        'loader': [ds_key]
                        }),
                    ignore_index = True,
                    )

            df_idood = df_idood._append(
                    pd.DataFrame({
                        'score value': _scores,
                        'score type': \
                                [score_name+' ID' for i in range(len(s_id))] + \
                                [score_name+' OOD' for i in range(len(s_ood))]
                        }),
                    ignore_index = True,
                    )

            # saves colors and linestyles
            cs_idood[score_name+' ID'] = colors[score_n]
            cs_idood[score_name+' OOD'] = colors[score_n]
            ls_idood[score_name+' ID'] = '--' 
            ls_idood[score_name+' OOd'] = '-'

        #--------------------
        # Plotting
        #--------------------

        # plotting IDs and OODs distribution
        ax = axs[loader_n] 
        p = sb.kdeplot(
                data = df_idood,
                ax = ax,
                x = 'score value',
                common_norm = False,
                hue = 'score type',
                palette = cs_idood,
                hue_order = list(cs_idood.keys()),
                clip = [0., 1.],
                alpha = 0.75,
                legend = loader_n == 0
                )

        # set up linestyles
        for ls, line in zip(list(ls_idood.values()), p.lines):
            line.set_linestyle(ls)
        
        # set legend linestyle
        if loader_n == 0:
            handles = p.legend_.legend_handles[::-1]
            for ls, h in zip(list(ls_idood.values()), handles):
                h.set_ls(ls)
                                                                                        
        ax.set_xlabel('Score')
        ax.set_ylabel('%')
        ax.title.set_text(f'{ds_key}')
        ax.grid(True)

    # Plot AUCs
    ax = axs[-1]
    sb.pointplot(
            data = aucs_df,
            ax = ax,
            x = 'loader',
            y = 'AUC',
            hue = 'score name',
            markersize = 8,
            palette = colors[0:len(scores[ood_loaders[0]])],
            alpha = 0.75,
            legend = True
            )
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.savefig((path/f'in_out_distribution{suffix}.png').as_posix(), dpi=300, bbox_inches='tight')
    plt.close()
    return 
    
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
                y_true[:n_samples] = 1

                fpr, tpr, thresholds = roc_curve(y_true, s)

                if ds_key == 'val':
                    fpr_target = 0.05
                    keep = np.where(fpr <= fpr_target)[0]
                    idx = keep[-1]
                    threshold_fpr = thresholds[idx]
                    axs[i].axvline(threshold_fpr, color='black', linestyle='--', label=f"th={threshold_fpr:.2f}")
                    axs[i].set_title(f'Val - FPR @ 5%')
                    axs[i].legend()
                elif ds_key == 'test':
                    idx_test = np.argmin(np.abs(thresholds - threshold_fpr))

                    fpr_at_threshold = fpr[idx_test]
                    axs[i].axvline(threshold_fpr, color='black', linestyle='--', label=f"th={threshold_fpr:.2f}")
                    axs[i].set_title(f'Test - FPR @ {fpr_at_threshold*100:.2f}%')
                else:
                    raise RuntimeError('Do not mess with other portion of the dataset just val and test')
                
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

    
def plot_confidence(**kwargs):
    '''
    Plot OKs and KOs distributions and confidences. Confidences are computed for a score threshold 'th', assuming values bellow 'th' are wrongly classified and above are correctly classified true positive and negative ('TP(th)' and 'TN(th)') are computed, so 'conf(th) = (TP(th)+TN(th))/ns'. 'th' is plotted from 0 to 'max_score'. 

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
    max_score = kwargs.get('max_score', 20)
    calib_bin = kwargs.get('calib_bin', 1)
    path = kwargs.get('path', None)
    verbose = kwargs.get('verbose', False)

    # parse arguments
    if path == None: 
        path = Path.cwd()
    else:
        path = Path(path)

    if loaders == None: loaders = list(scores.keys())

    fig, axs = plt.subplots(2, len(loaders)+1, sharex='none', sharey='none', figsize=(5*(len(loaders)+1), 5*2))
    
    colors = ['xkcd:cobalt', 'xkcd:bluish green', 'xkcd:light orange', 'xkcd:dark hot pink', 'xkcd:purplish']
    lines = ['--', '-']

    # save AUCs for plotting 
    aucs_df = pd.DataFrame()
    drop_df = pd.DataFrame()

    for loader_n, ds_key in enumerate(loaders):
        # save OKs and KOs and confidences for plotting
        df_okko = pd.DataFrame()
        cs_okko, ls_okko = {}, {} 
        df_conf = pd.DataFrame()
        cs_conf, ls_conf = {}, {} 

        for score_n, score_name in enumerate(scores[ds_key].keys()):
            _scores = scores[ds_key][score_name]
            results = cvs._dss[ds_key]['result'] 
            ns = _scores.shape[0] # number of samples

            s_oks = _scores[results == True]
            s_kos = _scores[results == False]

            sorted_pos, _ = torch.sort(s_oks, descending=True)
            sorted_neg, _ = torch.sort(s_kos, descending=True)
            if ds_key == 'test':
                tpr95_index = int(torch.ceil(torch.tensor(0.95 * sorted_pos.numel())).item()) - 1
                threshold = sorted_pos[tpr95_index]                
                fpr95 = (s_kos >= threshold).float().mean().item()
                print(f'FPR95 for {ds_key} {score_name} split: {fpr95:.4f}')

                fpr5_index = int(torch.ceil(torch.tensor(0.05 * sorted_neg.numel())).item()) - 1
                fpr5_index = max(0, min(fpr5_index, sorted_neg.numel() - 1))

                # threshold = punteggio del negativo a quel rank
                threshold = sorted_neg[fpr5_index]
               
                tpr5 = (s_oks >= threshold).float().mean().item()
                print(f'TPR5 for {ds_key} {score_name} split: {tpr5:.4f}')
        
            # compute AUC for score and model
            auc = AUC().update(_scores, results.int()).compute().item()
            if verbose: print(f'AUC for {ds_key} {score_name} split: {auc:.4f}')
            aucs_df = aucs_df._append(
                    pd.DataFrame({
                        'AUC': [auc],
                        'score name': [score_name],
                        'loader': [ds_key]
                        }),
                    ignore_index = True,
                    )
            
            df_okko = df_okko._append(
                    pd.DataFrame({
                        'score value': torch.hstack((s_oks, s_kos)),
                        'score type': \
                                [score_name+': OK' for i in range(len(s_oks))] + \
                                [score_name+': KO' for i in range(len(s_kos))]
                                }),
                    ignore_index = True,
                    )

            # saves colors and linestyles
            cs_okko[score_name+': OK'] = colors[score_n]
            cs_okko[score_name+': KO'] = colors[score_n]
            ls_okko[score_name+': OK'] = '--' 
            ls_okko[score_name+': KO'] = '-'
            
            # Compute accuracies
            s_acc = torch.zeros(int(100*max_score)+1)
            s_drop = torch.zeros(int(100*max_score)+1)
            ths = torch.zeros(int(100*max_score)+1)
            for i, th in enumerate(torch.linspace(0., max_score, int(100*max_score)+1)):
                ths[i] = th
                s_idx = _scores > th 
                s_acc[i] = (results[s_idx].sum() + results[s_idx.logical_not()].logical_not().sum())/ns
                s_drop[i] = (_scores <= th).sum()/ns

            df_conf = df_conf._append(
                    pd.DataFrame({
                        'value': 100*torch.hstack((s_acc, s_drop)),
                        'ths': ths.repeat(2),
                        'score type': \
                                [score_name+': Acc' for i in range(len(s_acc))] + \
                                [score_name+': Dropped' for i in range(len(s_drop))]
                        }),
                    ignore_index = True,
                    )

            # saves colors and linestyles
            cs_conf[score_name+': Acc'] = colors[score_n]
            cs_conf[score_name+': Dropped'] = colors[score_n]
            ls_conf[score_name+': Acc'] = '-' 
            ls_conf[score_name+': Dropped'] = '--' 

            # how many samples are dropped (smalled than th) at max acc
            drop_df = drop_df._append(
                    pd.DataFrame({
                        'Drop': [100*s_drop[s_acc.argmax()].item()],
                        'score name': [score_name],
                        'loader': [ds_key]
                        }),
                    ignore_index = True,
                    )

        #--------------------
        # Plotting
        #--------------------

        # plotting OKs and KOs distribution
        ax = axs[0][loader_n] 
        p = sb.kdeplot(
                data = df_okko,
                ax = ax,
                x = 'score value',
                common_norm = False,
                hue = 'score type',
                hue_order = list(cs_okko.keys()), 
                palette = cs_okko,
                clip = [0., 1.],
                alpha = 0.75,
                legend = loader_n == len(loaders)-1
                )

        # set up linestyles
        for ls, line in zip(list(ls_okko.values()), p.lines):
            line.set_linestyle(ls)
        
        # set legend linestyle
        if loader_n == len(loaders)-1:
            handles = p.legend_.legend_handles[::-1]
            for ls, h in zip(list(ls_okko.values()), handles):
                h.set_ls(ls)

        ax.set_xlabel('Score')
        ax.set_ylabel('%')
        ax.title.set_text(f'{ds_key}')
        ax.grid(True)

        # plotting Confidences
        ax = axs[1][loader_n]
        p = sb.lineplot(
                data = df_conf,
                ax = ax,
                x = 'ths', 
                y = 'value',
                hue = 'score type',
                hue_order = list(cs_conf.keys()), 
                palette = cs_conf,
                alpha = 0.75,
                legend = loader_n == len(loaders)-1,
                )

        # set up linestyles
        for ls, line in zip(list(ls_conf.values()), p.lines):
            line.set_linestyle(ls)
        
        # set legend linestyle
        if loader_n == len(loaders)-1:
            handles = p.legend_.legend_handles[::-1]
            for h in handles:
                h.set_ls(ls_conf[h._label])

        ax.set_xlabel('Score Threshold')
        ax.set_ylabel('%')
        ax.grid(True)

    # Plot AUCs
                                                              
    ax = axs[0][-1]
    sb.pointplot(
            data = aucs_df,
            ax = ax,
            x = 'loader',
            y = 'AUC',
            hue = 'score name',
            markersize = 8,
            palette = colors[0:len(scores[loaders[-1]])],
            alpha = 0.75,
            legend = True
            )
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # plot samples droppedn (<th) at max acc
    ax = axs[1][-1]
    sb.pointplot(
            data = drop_df,
            ax = ax,
            x = 'loader',
            y = 'Drop',
            hue = 'score name',
            markersize = 8,
            palette = colors[0:len(scores[loaders[-1]])],
            alpha = 0.75,
            legend = True
            )
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_ylabel('# score < th (%)')

    plt.savefig((path/f'confidence.png').as_posix(), dpi=300, bbox_inches='tight')
    plt.close()
    return 

def plot_calibration(**kwargs):
    '''
    Plot calibration curves.

    Args:
    - corevectors (peepholelib.coreVectors.CoreVectors): corevectors with dataset parsed (see `peepholelib.coreVectors.parse_ds`).
    - scores (dict(str:dict(str: torch.tensor))): Two-level dictionary with first keys being the loader name, seconde-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'scores'. Defaults to 'None'.
    - path ('str'): Path to save plots.
    - calib_bin (int): Bin size for calibration plot.
    - verbose (bool): print progress messages.
    '''

    cvs = kwargs.get('corevectors')
    scores = kwargs.get('scores')
    loaders = kwargs.get('loaders', None)
    calib_bin = kwargs.get('calib_bin', 0.1)
    path = kwargs.get('path', None)
    verbose = kwargs.get('verbose', False)

    # parse arguments
    if path == None: 
        path = Path.cwd()
    else:
        path = Path(path)
    print(path)

    if loaders == None: loaders = list(scores.keys())

    fig, axs = plt.subplots(1, len(loaders)+1, sharex='none', sharey='none', figsize=(5*(len(loaders)+1), 5))

    colors = ['xkcd:cobalt', 'xkcd:bluish green', 'xkcd:light orange', 'xkcd:dark hot pink', 'xkcd:purplish']

    n_bins = ceil(1/calib_bin)
    eces_df = pd.DataFrame()
    for loader_n, ds_key in enumerate(loaders):

        df_calib = pd.DataFrame()
        for score_name in scores[ds_key].keys():
            _scores = scores[ds_key][score_name]
            results = cvs._dss[ds_key]['result'] 
            ns = _scores.shape[0] # number of samples

            # comput calibration and ECEs
            s_acc = torch.zeros(n_bins)
            s_conf = torch.zeros(n_bins)
            s_ns  = torch.zeros(n_bins)

            for b in range(n_bins):
                s_idx = torch.logical_and(_scores>b*calib_bin, _scores<=(b+1)*calib_bin)
                s_ns[b] = s_idx.sum() # how many samples in the bin
                s_acc[b] = (results[s_idx]).sum()/s_ns[b] # bin's accuracy
                s_conf[b] = _scores[s_idx].sum()/s_ns[b] # bin's average confidence

            # avoid numerical problems with NaN
            s_acc = torch.nan_to_num(s_acc) 
            s_conf = torch.nan_to_num(s_conf) 

            # Compute ECE score
            ece = ((s_ns*(s_acc-s_conf).abs()).sum()/ns).item()
            if verbose: print(f'ECE for {ds_key} {score_name} split: {ece:.4f}')
            eces_df = eces_df._append(
                    pd.DataFrame({
                        'ECE': [ece],
                        'score name': [score_name],
                        'loader': [ds_key]
                        }),
                    ignore_index = True,
                    )

            df_calib = df_calib._append(
                    pd.DataFrame({
                        'accuracy': s_acc,
                        'score type': [score_name for i in range(n_bins)]
                        }),
                    ignore_index = True,
                    )
        
        # add perfect calibration
        x = torch.linspace(0, 1, n_bins+1)[:-1]
        df_perf_calib = pd.DataFrame({
            'x': x,
            'y': x, 
            'score type': ['Perfect Calibration' for i in range(n_bins)]
            })

        #--------------------
        # Plotting
        #--------------------
        ax = axs[loader_n]
        sb.pointplot(
                data = df_calib,
                ax = ax,
                x = x.repeat(len(scores[ds_key].keys())),
                y = 'accuracy',
                hue = 'score type',
                palette = colors[0:len(scores[loaders[-1]])],
                alpha = 0.75,
                markersize = 8,
                legend = loader_n == len(loaders)-1,
                )

        sb.pointplot(
                data = df_perf_calib,
                ax = ax,
                x = 'x',
                y = 'y',
                markersize = 0,
                hue = 'score type',
                palette = ['xkcd:red'],
                alpha = 0.75,
                legend = loader_n == len(loaders)-1,
                )

        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy (%)')
        ax.title.set_text(f'{ds_key}')
        ax.grid(True)
    
    # Plot ECEs 
    ax = axs[-1]
    sb.pointplot(
            data = eces_df,
            ax = ax,
            x = 'loader',
            y = 'ECE',
            hue = 'score name',
            markersize = 8,
            palette = colors[0:len(scores[loaders[-1]])],
            alpha = 0.75,
            legend = True
            )
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.savefig((path/f'calibration.png').as_posix(), dpi=300, bbox_inches='tight')
    plt.close()
    return 
