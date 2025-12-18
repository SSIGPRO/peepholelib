from pathlib import Path as Path

import numpy as np
import pandas as pd
import torch
from torcheval.metrics import BinaryAUROC as AUC


def _score_values(scores, loader, score_name):
    return torch.tensor(
            scores.loc[
                (scores['dataset'] == loader) & (scores['score name'] == score_name),
                'score value',
                ].tolist(),
            dtype=torch.float32,
            )


def _result_mask(dss, loader):
    results = dss._dss[loader]['result']
    if isinstance(results, torch.Tensor):
        return results.bool().view(-1)
    return torch.as_tensor(results, dtype=torch.bool).view(-1)


def eval_corruptions(**kwargs):
    '''
    Compute AUROC scores between clean ID samples and misclassified corruptions.

    Args:
    - scores (pandas.DataFrame): Score dataframe with columns 'dataset',
      'score name', and 'score value'.
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - id_loader (str): clean in-distribution loader.
    - c_loaders (list[str]): corruption loaders to consider.
    - score_names (str|list[str]): score name(s) to use. If None, all are tried.
    - id_correct_only (bool): keep only correctly classified ID samples.
      Defaults to True.

    Returns:
    - pandas.DataFrame: one row for each ID/corruption pair with the same score
      name.
    '''

    scores = kwargs['scores']
    dss = kwargs['datasets']
    id_loader = kwargs['id_loader']
    c_loaders = kwargs['c_loaders']
    score_names = kwargs.get('score_names', kwargs.get('score_name', None))
    id_correct_only = kwargs.get('id_correct_only', kwargs.get('filter_id_correct', True))
    verbose = kwargs.get('verbose', False)

    if isinstance(c_loaders, str):
        c_loaders = [c_loaders]

    if score_names is None:
        score_names = scores['score name'].drop_duplicates().tolist()
    elif isinstance(score_names, str):
        score_names = [score_names]

    rows = []
    for score_name in score_names:
        s_id = _score_values(scores, id_loader, score_name)

        if id_correct_only:
            id_results = _result_mask(dss, id_loader)
            if len(id_results) != len(s_id):
                raise ValueError(
                        f'ID result length mismatch for {id_loader} '
                        f'({len(id_results)} results, {len(s_id)} scores)'
                        )
            s_id = s_id[id_results == True]

        if len(s_id) == 0:
            if verbose:
                print(f'No ID samples available for {id_loader} {score_name}')
            continue

        for c_loader in c_loaders:
            s_corruption_all = _score_values(scores, c_loader, score_name)
            c_results = _result_mask(dss, c_loader)

            if len(c_results) != len(s_corruption_all):
                raise ValueError(
                        f'Corruption result length mismatch for {c_loader} '
                        f'({len(c_results)} results, {len(s_corruption_all)} scores)'
                        )

            s_corruption = s_corruption_all[c_results == False]

            if len(s_corruption) == 0:
                auc = np.nan
                if verbose:
                    print(f'No misclassified samples for {c_loader} {score_name}')
            else:
                labels = torch.hstack((
                    torch.ones(s_id.shape),
                    torch.zeros(s_corruption.shape),
                    ))
                values = torch.hstack((s_id, s_corruption))
                auc = AUC().update(values, labels).compute().item()

            rows.append({
                'score name': score_name,
                'id loader': id_loader,
                'corruption loader': c_loader,
                'AUC': auc,
                'n id': len(s_id),
                'n corruption': len(s_corruption),
                'n corruption all': len(s_corruption_all),
            })

    return pd.DataFrame(rows)


def print_corruptions_aucs(**kwargs):
    '''
    Print AUC scores for corruption loaders, keeping only misclassified
    corruption samples.

    Args:
    - aucs (pandas.DataFrame): optional output of eval_corruptions.
    - scores (pandas.DataFrame): score dataframe, required when aucs is omitted.
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed
      dataset, required when aucs is omitted.
    - id_loader (str): clean in-distribution loader.
    - c_loaders (list[str]): corruption loaders to analyze. If None, all in
      aucs are used.
    - csv_path (str): path of the CSV file to save. If None, no file is saved.
    '''

    aucs = kwargs.get('aucs', None)
    id_loader = kwargs['id_loader']
    c_loaders = kwargs.get('c_loaders', None)
    csv_path = kwargs.get('csv_path', None)

    if aucs is None:
        aucs = eval_corruptions(**kwargs)

    df = aucs.loc[aucs['id loader'] == id_loader].copy()

    if c_loaders is not None:
        if isinstance(c_loaders, str):
            c_loaders = [c_loaders]
        df = df.loc[df['corruption loader'].isin(c_loaders)]

    for c_loader, group in df.groupby('corruption loader'):
        print(f'Corruption loader: {c_loader}')
        for _, row in group.iterrows():
            name = row["score name"].removesuffix(f'-{c_loader}')
            print(f'  {name}: {row["AUC"]:.4f} (n={int(row["n corruption"])})')

    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    return df
