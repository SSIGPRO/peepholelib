# General pytho stuff
from pathlib import Path as Path

# plotting stuff
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sb
import pandas as pd

# torch stuff
import torch
from torcheval.metrics import BinaryAUROC as AUC

def plot_confidence(**kwargs):
    '''
    Plot OKs and KOs distributions and confidences. Confidences are computed for a score threshold 'th', assuming values bellow 'th' are wrongly classified and above are correctly classified true positive and negative ('TP(th)' and 'TN(th)') are computed, so 'conf(th) = (TP(th)+TN(th))/ns'. 'th' is plotted from 0 to 'max_score'. 

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - scores (dict(str:dict(str: torch.tensor))): Two-level dictionary with first keys being the loader name, seconde-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'scores'. Defaults to 'None'.
    - path ('str'): Path to save plots.
    - max_score (float): Max score for the accuracy plot, within '[0., 1.]'.
    - loaders_renames (list[str}): list of names to overwrite the loaders' names in the plots. 
    - verbose (bool): print progress messages.
    '''

    dss = kwargs.get('datasets')
    scores = kwargs.get('scores')
    loaders = kwargs.get('loaders', None)
    max_score = kwargs.get('max_score', 20)
    calib_bin = kwargs.get('calib_bin', 1)
    path = kwargs.get('path', None)
    loaders_renames = kwargs.get('loaders_renames', None)
    verbose = kwargs.get('verbose', False)

    # parse arguments
    if path == None: 
        path = Path.cwd()
    else:
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if loaders == None: loaders = list(scores.keys())

    fig, axs = plt.subplots(1, len(loaders), sharex='none', sharey='none', figsize=(5*(len(loaders)), 5))
    
    colors = ['xkcd:cobalt', 'xkcd:bluish green', 'xkcd:light orange', 'xkcd:dark hot pink', 'xkcd:purplish', 'xkcd:slate gray', 'xkcd:cinnamon']

    for loader_n, ds_key in enumerate(loaders):
        # save OKs and KOs and confidences for plotting
        df_okko = pd.DataFrame()
        cs_okko, ls_okko = {}, {} 

        for score_n, score_name in enumerate(scores[ds_key].keys()):
            _scores = scores[ds_key][score_name]
            results = dss._dss[ds_key]['result'] 
            ns = _scores.shape[0] # number of samples

            s_oks = _scores[results == True]
            s_kos = _scores[results == False]

            # compute AUC and TPR@95 for score and model
            auc = AUC().update(_scores, results.int()).compute().item()
            sorted_pos, _ = torch.sort(s_oks, descending=True)
            tpr95_index = int(torch.ceil(torch.tensor(0.95 * sorted_pos.numel())).item()) - 1
            threshold = sorted_pos[tpr95_index]                
            fpr95 = (s_kos >= threshold).float().mean().item()

            if verbose:
                print(f'FPR95 for {ds_key} {score_name} split: {fpr95:.4f}')
                print(f'AUC for {ds_key} {score_name} split: {auc:.4f}')

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
            
        #--------------------
        # Plotting
        #--------------------
        if len(loaders) == 1:
            ax = axs
        else:
            ax = axs[loader_n] 

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
                legend=False,  # <- important: we'll add custom legends
                )

        # set up linestyles
        for ls, line in zip(list(ls_okko.values()), p.lines):
            line.set_linestyle(ls)
        
        # -------------------------
        # Build 2 separate legends
        # -------------------------

        # 1) Color legend: score types (one color per score_name)
        score_names = list(scores[ds_key].keys())  # order matches 'colors'
        lw = 2.0
        color_handles = [
            Line2D([0], [0], color=colors[i], lw=lw)
            for i, _ in enumerate(score_names)
        ]
        color_legend = ax.legend(
            color_handles,
            score_names,
            title='Score type',
            loc='upper left',
            frameon=True,
        )

        # 2) Linestyle legend: outcome (OK vs KO)
        ls_handles = [
            Line2D([0], [0], color='black', lw=lw, linestyle='-'),  # OK
            Line2D([0], [0], color='black', lw=lw, linestyle='--'),   # KO
        ]
        ls_legend = ax.legend(
            ls_handles,
            ['correct', 'wrong'],
            title='Classification',
            loc='upper left',
            bbox_to_anchor=(0.25, 1.0),
            frameon=True,
        )

        # Make sure both legends show
        ax.add_artist(color_legend)

        # labels / grid
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        if loaders_renames is not None:
            ax.title.set_text(f'{loaders_renames[loader_n]}')
        else:
            ax.title.set_text(f'{ds_key}')
        ax.grid(True)

    plt.savefig((path/f'confidence.png').as_posix(), dpi=300, bbox_inches='tight')
    plt.close()
    return 

def one_thr_for_all(**kwargs):

    '''
    Unique evaluation for all cases together to see what happens

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - scores (dict(str:dict(str: torch.tensor))): Two-level dictionary with first keys being the loader name, seconde-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'scores'. Defaults to 'None'.
    - path ('str'): Path to save plots.
    - max_score (float): Max score for the accuracy plot, within '[0., 1.]'.
    - loaders_renames (list[str}): list of names to overwrite the loaders' names in the plots. 
    - verbose (bool): print progress messages.
    '''

    dss = kwargs.get('datasets')
    scores = kwargs.get('scores')
    scores_ids = kwargs.get('scores_ids', scores.keys())
    id_loader = kwargs.get('id_loader')
    c_loaders = kwargs.get('c_loaders')
    ood_loaders = kwargs.get('ood_loaders')
    atk_loaders = kwargs.get('atk_loaders')
    path = kwargs.get('path', None)
    verbose = kwargs.get('verbose', False)

    # # parse arguments
    # if path == None: 
    #     path = Path.cwd()
    # else:
    #     path = Path(path)
    # path.mkdir(parents=True, exist_ok=True)

    thrs = {}
    for score_name in scores_ids:
        _scores = scores[id_loader][score_name]
        results = dss._dss[id_loader]['result']

        s_oks = _scores[results == True]

        sorted_pos, _ = torch.sort(s_oks, descending=True)
        tpr95_index = int(torch.ceil(torch.tensor(0.95 * sorted_pos.numel())).item()) - 1
        thrs[score_name] = sorted_pos[tpr95_index] 

    print('-----------\n CORRUPTION \n-----------')

    for cl in c_loaders:

        for score_name in scores_ids:
 
            s_kos = scores[cl][score_name]### here should be a random permutation of the elements so that we have the same amount
               
            fpr95 = (s_kos >= thrs[score_name]).float().mean().item()

            if verbose:
                print(f'FPR95 for {cl} {score_name} split: {fpr95:.2f}')

    print('-----------\n OOD \n-----------')

    for ol in ood_loaders:

        for score_name in scores_ids:
            
            s_kos = scores[ol][score_name]### here should be a random permutation of the elements so that we have the same amount
            
            fpr95 = (s_kos >= thrs[score_name]).float().mean().item()

            if verbose:
                print(f'FPR95 for {ol} {score_name} split: {fpr95:.2f}')

    print('-----------\n ATTACKS \n-----------')

    for al in atk_loaders:

        for score_name in scores_ids:

            atk_success = dss._dss[al]['attack_success']
            s_kos = scores[al][score_name][atk_success == True]### here should be a random permutation of the elements so that we have the same amount
              
            fpr95 = (s_kos >= thrs[score_name]).float().mean().item()

            if verbose:
                print(f'FPR95 for {al} {score_name} split: {fpr95:.2f}')


