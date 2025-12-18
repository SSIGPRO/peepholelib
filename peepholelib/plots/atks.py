# General pytho stuff
from pathlib import Path as Path
from math import ceil

# plotting stuff
from sklearn.metrics import roc_curve, roc_auc_score, auc
import pandas as pd

# torch stuff
import torch
from torcheval.metrics import BinaryAUROC as AUC

def print_atks_aucs(**kwargs):
    '''
    Print AUC scores for all attack loaders of a given clean loader from eval_atks output.

    Args:
    - aucs (pandas.DataFrame): output of eval_atks, with columns 'score name',
      'ori loader', 'atk loader', 'AUC'.
    - ori_loader (str): clean dataset name to filter on.
    - atk_loaders (list[str]): attack datasets to analyze. If None, all are used.
    - csv_path (str): path of the CSV file to save. If None, no file is saved.
    '''
    aucs = kwargs['aucs']
    ori_loader = kwargs['ori_loader']
    atk_loaders = kwargs.get('atk_loaders', None)
    csv_path = kwargs.get('csv_path', None)

    df = aucs.loc[aucs['ori loader'] == ori_loader].copy()

    if atk_loaders is not None:
        df = df.loc[df['atk loader'].isin(atk_loaders)]

    for atk_loader, group in df.groupby('atk loader'):
        print(f'ATK loader: {atk_loader}')
        for _, row in group.iterrows():
            name = row["score name"].removesuffix(f'-{atk_loader}')
            print(f'  {name}: {row["AUC"]:.4f}')

    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    return df

def eval_corruptions(**kwargs):
    '''
    Compute AUC scores between clean/corrupted pairs, restricted to samples
    where the clean prediction was correct.

    Requires that clean and corrupted datasets are index-aligned (same sample
    ordering and size), which holds when both are parsed from the same source
    with the same seed.

    Args:
    - scores (pandas.DataFrame): dataframe with columns 'dataset', 'score name',
      and 'score value'.
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed datasets.
    - ori_loader (str): clean dataset loader name.
    - corr_loaders (list[str]): corruption loader names.
    - score_names (str|list[str]): score name(s) to use. If None, all are tried.
    - filter_correct (bool): if True (default), restrict to samples where the clean
      prediction was correct, using datasets._dss[ori_loader][:]['result'].

    Returns:
    - pandas.DataFrame: one row per clean/corrupted pair with the same score name.
    '''
    scores = kwargs['scores']
    dss = kwargs.get('datasets', None)
    ori_loader = kwargs['ori_loader']
    corr_loaders = kwargs['corr_loaders']
    score_names = kwargs.get('score_names', kwargs.get('score_name', None))
    filter_correct = kwargs.get('filter_correct', True)

    if score_names is None:
        score_names = scores['score name'].drop_duplicates().tolist()
    elif isinstance(score_names, str):
        score_names = [score_names]

    correct_mask = None
    if filter_correct and dss is not None:
        correct_mask = dss._dss[ori_loader][:]['result'].bool()

    rows = []
    for score_name in score_names:
        score_df = scores.loc[scores['score name'] == score_name]

        s_ori_all = score_df.loc[
            score_df['dataset'] == ori_loader,
            'score value',
        ].dropna().tolist()

        if len(s_ori_all) == 0:
            continue

        s_ori_all = torch.tensor(s_ori_all, dtype=torch.float32)

        for corr_loader in corr_loaders:
            s_corr_all = score_df.loc[
                score_df['dataset'] == corr_loader,
                'score value',
            ].dropna().tolist()

            if len(s_corr_all) == 0:
                continue

            s_corr_all = torch.tensor(s_corr_all, dtype=torch.float32)

            if correct_mask is not None and len(correct_mask) == len(s_ori_all) == len(s_corr_all):
                s_ori = s_ori_all[correct_mask]
                s_corr = s_corr_all[correct_mask]
            else:
                _ns = min(len(s_ori_all), len(s_corr_all))
                s_ori = s_ori_all[torch.randperm(len(s_ori_all))[:_ns]]
                s_corr = s_corr_all[torch.randperm(len(s_corr_all))[:_ns]]

            if len(s_ori) == 0 or len(s_corr) == 0:
                continue

            labels = torch.hstack((torch.ones(s_ori.shape), torch.zeros(s_corr.shape)))
            values = torch.hstack((s_ori, s_corr))
            auc_value = roc_auc_score(labels.numpy(), values.numpy())

            rows.append({
                'score name': score_name,
                'ori loader': ori_loader,
                'atk loader': corr_loader,
                'AUC': auc_value,
                'n ori': len(s_ori),
                'n corr': len(s_corr),
            })

    return pd.DataFrame(rows)


def eval_atks(**kwargs):
    '''
    Compute AUC scores between clean/attacked pairs with the same score name.

    Args:
    - scores (pandas.DataFrame): dataframe with columns 'dataset', 'score name',
      and 'score value'.
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed datasets.
    - ori_loader (str): clean (original) dataset name.
    - atk_loaders (list[str]): attack loaders to consider. If None, all loaders
      in scores that are not ori_loader are used.
    - filter_key (str): key in datasets._dss[atk_loader] to filter successful
      attacks. If None, no filtering is applied. Defaults to 'attack_success'.
    - score_names (str|list[str]): score name(s) to use. If None, all are tried.

    Returns:
    - pandas.DataFrame: one row for each ori/atk pair with the same score name.
    '''
    scores = kwargs['scores']
    dss = kwargs.get('datasets', None)
    ori_loader = kwargs['ori_loader']
    atk_loaders = kwargs['atk_loaders']
    filter_key = kwargs.get('filter_key', 'attack_success')
    score_names = kwargs.get('score_names', kwargs.get('score_name', None))

    if score_names is None:
        score_names = scores['score name'].drop_duplicates().tolist()
    elif isinstance(score_names, str):
        score_names = [score_names]

    rows = []
    for score_name in score_names:
        score_df = scores.loc[scores['score name'] == score_name]

        s_ori_all = score_df.loc[
            score_df['dataset'] == ori_loader,
            'score value',
        ].dropna().tolist()

        if len(s_ori_all) == 0:
            continue

        for atk_loader in atk_loaders:
            s_atk_all = score_df.loc[
                score_df['dataset'] == atk_loader,
                'score value',
            ].dropna().tolist()

            if len(s_atk_all) == 0:
                continue

            s_ori = torch.tensor(s_ori_all, dtype=torch.float32)
            s_atk = torch.tensor(s_atk_all, dtype=torch.float32)

            if filter_key is not None and dss is not None:
                idx = dss._dss[atk_loader][:][filter_key] == 1
                s_ori = s_ori[idx]
                s_atk = s_atk[idx]
            else:
                _ns = min(len(s_ori), len(s_atk))
                s_ori = s_ori[torch.randperm(len(s_ori))[:_ns]]
                s_atk = s_atk[torch.randperm(len(s_atk))[:_ns]]

            if len(s_ori) == 0 or len(s_atk) == 0:
                continue

            labels = torch.hstack((torch.ones(s_ori.shape), torch.zeros(s_atk.shape)))
            values = torch.hstack((s_ori, s_atk))
            auc_value = roc_auc_score(labels.numpy(), values.numpy())

            rows.append({
                'score name': score_name,
                'ori loader': ori_loader,
                'atk loader': atk_loader,
                'AUC': auc_value,
                'n ori': len(s_ori),
                'n atk': len(s_atk),
            })

    return pd.DataFrame(rows)
