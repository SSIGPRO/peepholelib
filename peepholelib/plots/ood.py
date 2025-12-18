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

def eval_ood(**kwargs):
    '''
    Compute OOD AUROC scores between ID/OOD pairs with the same score name.

    Args:
    - scores (pandas.DataFrame): dataframe with columns 'dataset', 'score name',
      and 'score value'.
    - id_loader (str): in-distribution dataset name. Defaults to
      'ImageNet-test-ViTB16'.
    - split (str): split name to keep in the dataset column. Defaults to
      'test'. Use None to disable this filter.
    - score_names (str|list[str]): score name(s) to use. If None, all score
      names are tried.

    Returns:
    - pandas.DataFrame: one row for each ID/OOD pair with the same score name.
    '''
    scores = kwargs['scores']
    id_loader = kwargs.get('id_loader', 'ImageNet-test-ViTB16')
    split = kwargs.get('split', 'test')
    score_names = kwargs.get('score_names', kwargs.get('score_name', None))
    if split is not None:
        scores = scores.loc[
            scores['dataset'].str.contains(split, case=False, na=False)
        ]

    if score_names is None:
        score_names = scores['score name'].drop_duplicates().tolist()
    elif isinstance(score_names, str):
        score_names = [score_names]

    rows = []
    for score_name in score_names:
        score_df = scores.loc[scores['score name'] == score_name]

        s_id = score_df.loc[
            score_df['dataset'] == id_loader,
            'score value',
        ].dropna().tolist()

        ood_loaders = [
            ds for ds in score_df['dataset'].drop_duplicates().tolist()
            if ds != id_loader
        ]

        for ood_loader in ood_loaders:
            s_ood = score_df.loc[
                score_df['dataset'] == ood_loader,
                'score value',
            ].dropna().tolist()

            labels = [1] * len(s_id) + [0] * len(s_ood)
            values = s_id + s_ood
            auc_value = roc_auc_score(labels, values)

            rows.append({
                'score name': score_name,
                'id loader': id_loader,
                'ood loader': ood_loader,
                'AUC': auc_value,
                'n id': len(s_id),
                'n ood': len(s_ood),
            })

    return pd.DataFrame(rows)

def plot_ood(**kwargs):
    '''
    Plot OOD detection.

    Args:
    - scores (pandas.DataFrame): Score dataframe with columns 'dataset', 'score name', and 'score value'.
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
                id_loader = _id_loader[loader_n]
            else:
                id_loader = _id_loader

            s_id = torch.tensor(
                    scores.loc[
                        (scores['dataset'] == id_loader) & (scores['score name'] == score_name),
                        'score value',
                        ].tolist(),
                    dtype=torch.float32,
                    )
            s_ood = torch.tensor(
                    scores.loc[
                        (scores['dataset'] == ds_key) & (scores['score name'] == score_name),
                        'score value',
                        ].tolist(),
                    dtype=torch.float32,
                    )

            # guarantees the same number of samples
            _ns = min(len(s_id), len(s_ood))
            s_id = s_id[torch.randperm(len(s_id))[:_ns]]
            s_ood = s_ood[torch.randperm(len(s_ood))[:_ns]]

            # guarantees the same number of samples
            _ns = min(len(s_id), len(s_ood))
            s_id = s_id[torch.randperm(len(s_id))[:_ns]]
            s_ood = s_ood[torch.randperm(len(s_ood))[:_ns]]

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
            ls_idood[score_name+' OOD'] = '-'

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
            palette = colors[0:aucs_df['score name'].nunique()],
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

    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    # same palette length as number of score types
    n_scores = aucs_df['score name'].nunique()
    colors = ['xkcd:cobalt', 'xkcd:bluish green', 'xkcd:light orange',
              'xkcd:dark hot pink', 'xkcd:purplish', 'xkcd:slate gray',
              'xkcd:cinnamon', 'xkcd:azure'][:n_scores]

    sb.pointplot(
        data=aucs_df,
        ax=ax,
        x='loader',
        y='AUC',
        hue='score name',
        markersize=8,
        palette=colors,
        alpha=0.75,
        legend=True
    )

    if loaders_renames is not None:
        ax.set_xticks(range(len(loaders_renames)))
        ax.set_xticklabels(labels=loaders_renames, rotation=45, ha='right')
    else:
        ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45, ha='right')

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylabel('AUC')
    ax.set_xlabel('Loader')
    ax.legend_.remove()

    plt.tight_layout()
    plt.savefig(path / f'auc_only{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return

def print_ood_aucs(**kwargs):
    '''
    Print AUC scores for all OOD loaders of a given ID loader from score_ood output.

    Args:
    - aucs (pandas.DataFrame): output of score_ood, with columns 'score name',
      'id loader', 'ood loader', 'AUC'.
    - id_loader (str): in-distribution dataset name to filter on.
    - ood_loaders (list[str]): OOD datasets to analyze. If None, all are used.
    - csv_path (str): path of the CSV file to save. If None, no file is saved.
    '''
    aucs = kwargs['aucs']
    id_loader = kwargs['id_loader']
    ood_loaders = kwargs.get('ood_loaders', None)
    csv_path = kwargs['csv_path']

    df = aucs.loc[aucs['id loader'] == id_loader].copy()

    if ood_loaders is not None:
        df = df.loc[df['ood loader'].isin(ood_loaders)]

    for ood_loader, group in df.groupby('ood loader'):
        print(f'OOD loader: {ood_loader}')
        for _, row in group.iterrows():
            name = row["score name"].removesuffix(f'-{ood_loader}')
            print(f'  {name}: {row["AUC"]:.4f}')

    df.to_csv(csv_path, index=False)
