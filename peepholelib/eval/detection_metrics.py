# General python stuff
from math import log

# metrics stuff
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

def detection_metrics(**kwargs):
    '''
    Compute detection metrics separating the scores of a positive loader from the ones of each negative loader. The same computation applies to OOD and to adversarial attacks, and to the three families of scores: the ones which do not fit, the ones fitted on a single loader, and the ones calibrated against a negative loader.

    The scores calibrated against a negative loader hold it in the `'calib key'` column, so the positive samples selected are the ones scored for that calibration. For all the other scores `'calib key'` is `None` and the same positive scores are used against every negative loader.

    The metrics are computed over all the available samples, so the positive and the negative loaders do not need to have the same number of samples: `'AUROC'` and `'FPR@95'` are prevalence-independent, and `'AURC'` is normalized by `'E-AURC'`.

    Args:
    - scores (list[peepholelib.scores.score.Score]): scores to evaluate.
    - pos_loader (str): loader holding the positive samples.
    - neg_loaders (list[str]): loaders holding the negative samples.
    - metrics (list[str]): metrics to compute, among `'AUROC'`, `'FPR@95'`, `'AURC'` and `'E-AURC'`. If `None`, computes all of them. Defaults to `None`.
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed datasets, only used when `filter_key` is not `None`.
    - filter_key (str): key within `datasets._dss[<negative loader>]` used to keep a subset of the samples, e.g. `'attack_success'` to keep the successful attacks only. The same samples are kept on both sides, so it assumes the positive and the negative loaders hold the same samples in the same order, which only holds for attacks. If `None`, all samples are used. Defaults to `None`.
    - verbose (bool): print progress messages.

    Returns:
    - pandas.DataFrame: long-format metrics, with columns `['score name', 'pos loader', 'neg loader', 'n pos', 'n neg', 'metric', 'value']`.

    TODO: add `'AUPR-In'` and `'AUPR-Out'`, normalized by the prevalence `n_pos/(n_pos + n_neg)`.
    '''
    scores = kwargs['scores']
    pos_key = kwargs['pos_loader']
    neg_keys = kwargs['neg_loaders']
    metrics = kwargs.get('metrics')
    dss = kwargs.get('datasets')
    filter_key = kwargs.get('filter_key')
    verbose = kwargs.get('verbose', False)

    def risk_coverage(**kwargs):
        '''
        Risk of the accepted samples at each coverage, accepting the samples by decreasing score.
        '''
        values = kwargs['values']
        labels = kwargs['labels']

        order = values.argsort()[::-1]

        return (1 - labels[order]).cumsum()/np.arange(1, len(labels)+1)

    def auroc(**kwargs):
        '''
        Area under the ROC curve.
        '''
        return float(auc(kwargs['fpr'], kwargs['tpr']))

    def fpr_at_95(**kwargs):
        '''
        Fraction of negative samples scored above the threshold of the first operating point reaching at least 95% TPR. Tied scores move that point up, so the TPR it is taken at can exceed 95%.
        '''
        fpr = kwargs['fpr']
        tpr = kwargs['tpr']

        idx = min(int(np.searchsorted(tpr, 0.95, side='left')), len(fpr)-1)

        return float(fpr[idx])

    def aurc(**kwargs):
        '''
        Area under the risk-coverage curve.
        '''
        return float(risk_coverage(**kwargs).mean())

    def e_aurc(**kwargs):
        '''
        Excess AURC, the AURC of an oracle accepting all the positive samples first subtracted from the AURC. Removes the dependency on the fraction of negative samples.
        '''
        labels = kwargs['labels']

        error_rate = float(1 - labels.mean())
        optimal = error_rate + (1 - error_rate)*log(max(1 - error_rate, 1e-10))

        return aurc(**kwargs) - optimal

    # metrics available, keyed by name
    metric_fns = {
            'AUROC': auroc,
            'FPR@95': fpr_at_95,
            'AURC': aurc,
            'E-AURC': e_aurc,
            }

    metrics = metrics or list(metric_fns.keys())

    unknown = [m for m in metrics if m not in metric_fns]
    if len(unknown) != 0:
        raise ValueError(f'Unknown metrics {unknown}. Available metrics: {list(metric_fns.keys())}')

    rows = []
    for score in scores:
        _df = score._df

        for neg_key in neg_keys:
            calib = _df['calib key'].isna() | (_df['calib key'] == neg_key)

            s_pos = _df[calib & (_df['loader'] == pos_key)]['score value'].to_numpy(dtype=np.float64)
            s_neg = _df[calib & (_df['loader'] == neg_key)]['score value'].to_numpy(dtype=np.float64)

            # the score was not computed for this pair, e.g. a calibrated score with no calibration against neg_key
            if len(s_pos) == 0 or len(s_neg) == 0:
                if verbose: print('Skipping', score.name, 'for dataset', neg_key, ': no scores found')
                continue

            if filter_key is not None:
                idx = (dss._dss[neg_key][:][filter_key] == 1).cpu().numpy()
                s_pos = s_pos[idx]
                s_neg = s_neg[idx]

            values = np.concatenate((s_pos, s_neg))
            labels = np.concatenate((np.ones(len(s_pos)), np.zeros(len(s_neg))))
            fpr, tpr, _ = roc_curve(labels, values)

            _metrics = {m: metric_fns[m](values=values, labels=labels, fpr=fpr, tpr=tpr) for m in metrics}

            if verbose:
                _values = ', '.join(f'{m}: {v:.4f}' for m, v in _metrics.items())
                print(f'{score.name} for {pos_key} against {neg_key} ({len(s_pos)}/{len(s_neg)} samples): {_values}')

            rows += [{
                'score name': score.name,
                'pos loader': pos_key,
                'neg loader': neg_key,
                'n pos': len(s_pos),
                'n neg': len(s_neg),
                'metric': m,
                'value': v,
                } for m, v in _metrics.items()]

    return pd.DataFrame(rows)
