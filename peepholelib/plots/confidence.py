# General pytho stuff
from pathlib import Path as Path

# plotting stuff
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# torch stuff
import torch
from torcheval.metrics import BinaryAUROC as AUC

def eval_confidence(**kwargs):
    '''
    Compute and plot AUC and FPR@95 for each score and loader.

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - scores (pandas.DataFrame): Score dataframe with columns 'dataset', 'score name', and 'score value'.
    - score_names (str|list[str]): score name(s) to analyse. If 'None', gets all score names in 'scores'. Defaults to 'None'. 'score_name' is also accepted.
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'scores'. Defaults to 'None'.
    - path ('str'): Path to save plots.
    - loaders_renames (list[str}): list of names to overwrite the loaders' names in the plots.
    - verbose (bool): print progress messages.

    Returns:
    - results_df (pandas.DataFrame): DataFrame with columns 'dataset', 'score name', 'AUC', 'FPR@95'.
    '''

    dss = kwargs.get('datasets')
    scores = kwargs.get('scores')
    score_names = kwargs.get('score_names', kwargs.get('score_name', None))
    loaders = kwargs.get('loaders', None)
    path = kwargs.get('path', None)
    loaders_renames = kwargs.get('loaders_renames', None)
    verbose = kwargs.get('verbose', False)
    
    # parse arguments
    if path == None: 
        path = Path.cwd()
    else:
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if loaders is None:
        loaders = scores['dataset'].drop_duplicates().tolist()

    colors = ['xkcd:cobalt', 'xkcd:bluish green', 'xkcd:light orange', 'xkcd:dark hot pink', 'xkcd:purplish', 'xkcd:slate gray', 'xkcd:cinnamon']

    records = []
    for ds_key in loaders:
        print(f'evaluation of {ds_key}')
        for score_name in score_names:
            _scores = torch.tensor(
                    scores.loc[
                        (scores['dataset'] == ds_key) & (scores['score name'] == score_name),
                        'score value',
                        ].tolist(),
                    dtype=torch.float32,
                    )
            results = dss._dss[ds_key]['result']

            s_oks = _scores[results == True]
            s_kos = _scores[results == False]

            auc = AUC().update(_scores, results.int()).compute().item()
            sorted_pos, _ = torch.sort(s_oks, descending=True)
            tpr95_index = int(torch.ceil(torch.tensor(0.95 * sorted_pos.numel())).item()) - 1
            threshold = sorted_pos[tpr95_index]
            fpr95 = (s_kos >= threshold).float().mean().item()

            records.append({'dataset': ds_key, 'score name': score_name, 'AUC': auc, 'FPR@95': fpr95})
            print(f'{score_name}: AUC={auc:.4f}  FPR@95={fpr95:.4f}')

    results_df = pd.DataFrame(records)

    # --- grouped bar chart ---
    loader_labels = [loaders_renames[i] if loaders_renames is not None else loaders[i] for i in range(len(loaders))]
    n_loaders = len(loaders)
    n_scores = len(score_names)
    x = np.arange(n_loaders)
    width = 0.8 / n_scores

    fig, axs = plt.subplots(1, 2, figsize=(5 * 2, 5))

    for metric_idx, (metric, ax) in enumerate([('AUC', axs[0]), ('FPR@95', axs[1])]):
        for score_idx, score_name in enumerate(score_names):
            vals = [
                results_df.loc[(results_df['dataset'] == ds_key) & (results_df['score name'] == score_name), metric].values[0]
                for ds_key in loaders
            ]
            offset = (score_idx - (n_scores - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=score_name, color=colors[score_idx % len(colors)])

        ax.set_xticks(x)
        ax.set_xticklabels(loader_labels, rotation=15, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(title='Score', fontsize=8)
        ax.grid(True, axis='y')
        if metric == 'AUC':
            ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig((path / 'confidence.png').as_posix(), dpi=300, bbox_inches='tight')
    plt.close()
    return results_df

def one_thr_for_all(**kwargs):

    '''
    Unique evaluation for all cases together to see what happens

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - scores (pandas.DataFrame): Score dataframe with columns 'dataset', 'score name', and 'score value'.
    - scores_ids (str|list[str]): score name(s) to analyse. If 'None', gets all score names in the ID loader. Defaults to 'None'. 'score_names' and 'score_name' are also accepted.
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'], if 'None', gets all loaders in 'scores'. Defaults to 'None'.
    - path ('str'): Path to save plots.
    - max_score (float): Max score for the accuracy plot, within '[0., 1.]'.
    - loaders_renames (list[str}): list of names to overwrite the loaders' names in the plots. 
    - verbose (bool): print progress messages.
    '''

    dss = kwargs.get('datasets')
    scores = kwargs.get('scores')
    id_loader = kwargs.get('id_loader')
    c_loaders = kwargs.get('c_loaders')
    ood_loaders = kwargs.get('ood_loaders')
    atk_loaders = kwargs.get('atk_loaders')
    verbose = kwargs.get('verbose', False)
    scores_ids = kwargs.get('score_names', kwargs.get('score_name', None))

    thrs = {}
    fpr95 = {}

    for score_name in scores_ids:

        if not score_name in fpr95:
                fpr95[score_name] = {}

        _scores = torch.tensor(
                scores.loc[
                    (scores['dataset'] == id_loader) & (scores['score name'] == score_name),
                    'score value',
                    ].tolist(),
                dtype=torch.float32,
                )
        results = dss._dss[id_loader]['result']

        s_oks = _scores[results == True]
        s_kos = _scores[results == False]

        sorted_pos, _ = torch.sort(s_oks, descending=True)
        tpr95_index = int(torch.ceil(torch.tensor(0.95 * sorted_pos.numel())).item()) - 1
        thrs[score_name] = sorted_pos[tpr95_index] 
        fpr95[score_name][id_loader] = (s_kos >= thrs[score_name]).float().mean().item()

    print('-----------\n CORRUPTION \n-----------')

    for cl in c_loaders:

        for score_name in scores_ids:

            if not score_name in fpr95:
                fpr95[score_name] = {}
 
            s_kos = torch.tensor(
                    scores.loc[
                        (scores['dataset'] == cl) & (scores['score name'] == score_name),
                        'score value',
                        ].tolist(),
                    dtype=torch.float32,
                    )
            fpr95[score_name][cl] = (s_kos >= thrs[score_name]).float().mean().item()

            if verbose:
                print(f'FPR95 for {cl} {score_name} split: {fpr95[score_name][cl]:.2f}')

    print('-----------\n OOD \n-----------')

    for ol in ood_loaders:

        for score_name in scores_ids:

            if not score_name in fpr95:
                fpr95[score_name] = {}
            
            s_kos = torch.tensor(
                    scores.loc[
                        (scores['dataset'] == ol) & (scores['score name'] == score_name),
                        'score value',
                        ].tolist(),
                    dtype=torch.float32,
                    )
            
            fpr95[score_name][ol] = (s_kos >= thrs[score_name]).float().mean().item()

            if verbose:
                print(f'FPR95 for {ol} {score_name} split: {fpr95[score_name][ol]:.2f}')

    print('-----------\n ATTACKS \n-----------')

    for al in atk_loaders:

        for score_name in scores_ids:

            if not score_name in fpr95:
                fpr95[score_name] = {}

            atk_success = dss._dss[al]['attack_success']

            s_kos = torch.tensor(
                    scores.loc[
                        (scores['dataset'] == al) & (scores['score name'] == score_name),
                        'score value',
                        ].tolist(),
                    dtype=torch.float32,
                    )[atk_success == True]
              
            fpr95[score_name][al] = (s_kos >= thrs[score_name]).float().mean().item()

            if verbose:
                print(f'FPR95 for {al} {score_name} split: {fpr95[score_name][al]:.2f}')

    for score_name, v in fpr95.items(): 

        print(f'--------\n {score_name} \n--------')
        
        vals = torch.tensor(list(v.values()), dtype=torch.float32)
        print(f"overall FPR@95 = {vals.mean():.4f}")

        vals = torch.tensor(
            [val for k, val in v.items() if k != id_loader],
            dtype=torch.float32
        )
        print(f"Overall excluded ID FPR@95 = {vals.mean():.4f}")

        vals = torch.tensor([v[k] for k in c_loaders], dtype=torch.float32)
        print(f"corruptions FPR@95 = {vals.mean():.4f}")

        vals = torch.tensor([v[k] for k in ood_loaders], dtype=torch.float32)
        print(f"OOD FPR@95 = {vals.mean():.4f}")

        vals = torch.tensor([v[k] for k in atk_loaders], dtype=torch.float32)
        print(f"Attacks FPR@95 = {vals.mean():.4f}")

        

    
