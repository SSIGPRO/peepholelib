# General pytho stuff
from pathlib import Path as Path

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

def plot_ood(**kwargs):
    '''
    Plot OOD detection.

    Args:
    - scores (dict(str:dict(str: torch.tensor))): Two-level dictionary with first keys being the loader name, seconde-level key the score names and values the scores (see peepholelib.utils.scores.py). 
    - id_loaders (dict(str:str|list(str))): Dictionary of loaders of in-distribution data, with the key being the score type and values a str or list of strings for respective loaders.
    - ood_loaders (list[str]): out-of-distribution loaders to consider

    - path ('str'): Path to save plots.
    - suffix ('str'): Suffix to append to the plot's file name.
    - loaders_renames (list[str}): list of names to overwrite the loaders' names in the plots. 
    - verbose (bool): print progress messages.
    '''
    scores = kwargs.get('scores')
    id_loaders = kwargs.get('id_loaders')
    ood_loaders = kwargs.get('ood_loaders')
    path = kwargs.get('path', None)
    suffix = kwargs.get('suffix', '')
    loaders_renames = kwargs.get('loaders_renames', None)
    verbose = kwargs.get('verbose', False)

    # parse arguments
    if path == None: 
        path = Path.cwd()
    else:
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, len(ood_loaders)+1, sharex='none', sharey='none', figsize=(5*(len(ood_loaders)+1), 5))

    colors = ['xkcd:cobalt', 'xkcd:bluish green', 'xkcd:light orange', 'xkcd:dark hot pink', 'xkcd:purplish', 'xkcd:slate gray', 'xkcd:cinnamon', 'xkcd:azure' ]
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
                s_id = scores[_id_loader[loader_n]][score_name]
            else:
                s_id = scores[_id_loader][score_name]

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
                legend = False
                )

        # set up linestyles
        for ls, line in zip(list(ls_idood.values()), p.lines):
            line.set_linestyle(ls)
        
        for lbl, line in zip(list(cs_idood.keys()), p.lines):
            # lbl ends with ' ID' or ' OOD'
            if lbl.endswith(' ID'):
                line.set_linestyle('--')
            else:
                line.set_linestyle('-')
    
        # --- custom legends only on the first panel                                    
        if loader_n == 0:
            # METHODS (colors)
            # extract unique method names (strip " ID"/" OOD")
            all_labels = list(cs_idood.keys())
            methods = []
            for k in all_labels:
                name = k.replace(' ID', '').replace(' OOD', '')
                if name not in methods:
                    methods.append(name)
                                                                                        
            color_map = {m: colors[i] for i, m in enumerate(methods)}
            lw = 2.0
            method_handles = [Line2D([0], [0], color=color_map[m], lw=lw, linestyle='-', label=m) for m in methods]

            # CASES (line styles)
            case_handles = [
                Line2D([0], [0], color='k', lw=lw, linestyle='-',  label='ID'),
                Line2D([0], [0], color='k', lw=lw, linestyle='--', label='OOD'),
            ]
                                                                                        
            leg1 = ax.legend(
                    handles=method_handles,
                    title='Method',
                    loc='upper left'
                    )
            leg2 = ax.legend(
                    handles=case_handles,
                    title='Case',
                    loc='upper left',
                    bbox_to_anchor=(0.3, 1.0)
                    )
            ax.add_artist(leg1)  # keep both legends
            ax.set_ylabel('Density')

        ax.set_xlabel('Score')
        if loaders_renames is not None:
            ax.title.set_text(f'{loaders_renames[loader_n]}')
        else:
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

    if loaders_renames is not None:
        ax.set_xticks(range(len(loaders_renames)))
        ax.set_xticklabels(labels=loaders_renames)
    else:
        ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.savefig((path/f'in_out_distribution{suffix}.png').as_posix(), dpi=300, bbox_inches='tight')
    plt.close()
    return 
