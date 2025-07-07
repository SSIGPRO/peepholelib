# General pytho stuff
from tqdm import tqdm
from math import floor, ceil
from pathlib import Path as Path

# plotting stuff
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn.metrics import roc_curve, auc

# torch stuff
import torch
from torch.distributions import Categorical
from torcheval.metrics import BinaryAUROC as AUC
from torch.nn.functional import softmax as sm


def compute_top_k_accuracy(peepholes, targets, k):
    """
    Compute the topk accuracy given classification probabilities (shape 'n_samples*n_labels').
    
    Args:
    - peepholes (torch.tensor): probabilities of each sample to belong to each label
    - targets (torch.tensor): labels for all samples
    - k (int): top-k to compute
    """
    topk = torch.topk(peepholes, k=k, dim=1).indices
    targets = targets.unsqueeze(1)
    correct = (topk == targets).any(dim=1).float()
    return correct.mean().item()

def conceptogram_protoclass_score(**kwargs):
    '''
    Compute the Proto-Class score of all conceptograms in `phs._phs[`loaders`]`. `target_modules` are passed to `ph.get_conceptograms()` so the evaluation only consider the indicated modules. The score is computed by comparing the conceptogram with the protoclasses. #TODO: Add paper or a full description.
    If `plot=True` is passed, saves a KDE plot of the score and model confidence (`torch.nn.softmax(<model output>)`) for the correct and misclassified samples. `<model output>` is taken from `cvs`; it also plots the accuracy of the model dropping percentages of the dataset. 

    Args:
    - phs (peepholelib.peepholes.Peepholes): peepholes from which to take the conceptograms.
    - cvs (peepholelib.coreVectors.CoreVectors): corevectors respective to the `phs`.
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'].
    - target_modules (list[str]): list if target modules, as keys from the model `statedict`.
    - proto_key (str): the key in `loaders` to get compute the protoclasses from.
    - plot (bool): save figure with distributions of correctly and miss-classified samples.
    - max_drop (int): Max dataset drop for the accuracy plot.
    - calib_bin (int): Bin size for calibration plot.
    - verbose (bool): print progress messages.

    Returns:
    - ret (dict('score': dict(), 'auc':dict(), 'protoclasses':torch.tensor)): entropy scores and AUC for each values in `loaders`, and `protoclasses` for all classes with `shape=[n_model, n_modules, n_model]`, where `n_model` is the number of labels and `n_modules` the number of modules in `target_modules`.  
    '''

    phs = kwargs.get('peepholes')
    cvs = kwargs.get('corevectors')
    loaders = kwargs.get('loaders', ['test'])
    target_modules = kwargs.get('target_modules', None)
    proto_key = kwargs.get('proto_key', 'train')
    proto_th = kwargs.get('proto_threshold', 0.9)
    plot = kwargs.get('plot', False)
    drop_max = kwargs.get('max_drop', 20)
    calib_bin = kwargs.get('calib_bin', 1)
    verbose = kwargs.get('verbose', False)
    path = kwargs.get('path',None)

    score_name = 'Proto-Class'

    # get conceptogram 
    cpss = phs.get_conceptograms(loaders=loaders, target_modules=target_modules, verbose=verbose)

    if plot:
        fig, axs = plt.subplots(3, len(loaders), sharex='none', sharey='none', figsize=(4*len(loaders), 8))

    # sizes just to facilitate 
    nd = cpss[loaders[0]].shape[1] # number of layers (distributions)
    nc = cpss[loaders[0]].shape[2] # number of classes

    # compute proto-classes
    cps = cpss[proto_key]
    results = cvs._dss[proto_key]['result']
    labels = cvs._dss[proto_key]['label']
    confs = sm(cvs._dss[proto_key]['output'], dim=-1).max(dim=-1).values
    
    proto = torch.zeros(nc, nd, nc)
    for i in range(nc):
        cl = torch.logical_and(labels == i, results == 1)
        idx = torch.logical_and(cl, confs>proto_th)
        
        _p = cps[idx].sum(dim=0)  ## P'_j
        _p /= _p.sum(dim=1, keepdim=True)
        proto[i][:] = _p[:]

    # prepare returns dict
    ret = {
        'score': {},
        'protoclasses': proto,
    }

    # for normalization
    _line = torch.zeros(nc)
    _line[0] = 1
    min_e = Categorical(probs=_line).entropy()
    _unif = torch.ones(nc)/nc
    max_e = Categorical(probs=_unif).entropy()
    
    for loader_n, ds_key in enumerate(loaders):
        if path is None: path = Path.cwd()

        cps = cpss[ds_key]
        ns = cps.shape[0] # number of samples
        results = cvs._dss[ds_key]['result']
        #labels = (cvs._dss[ds_key]['label']).int()
        pred = (cvs._dss[ds_key]['pred']).int()
        confs = sm(cvs._dss[ds_key]['output'], dim=-1).max(dim=-1).values

        # main computation
        scores = (proto[pred]*cps).sum(dim=(1,2))
        scores = scores/(proto[pred].norm(dim=(1,2))*cps.norm(dim=(1,2)))

        ret['score'][ds_key] = scores 
        
        # plotting
        if plot:
            s_oks = scores[results == True]
            s_kos = scores[results == False]
            m_oks = confs[results == True]
            m_kos = confs[results == False]
            
            # compute AUC for score and model
            s_auc = AUC().update(scores, results.int()).compute().item()
            m_auc = AUC().update(confs, results.int()).compute().item()
            if verbose: print(f'AUC for {ds_key} split: {s_auc:.4f}')
            
            df = pd.DataFrame({
                'Value': torch.hstack((s_oks, s_kos, m_oks, m_kos)),
                'Score': \
                        [score_name+': OK' for i in range(len(s_oks))] + \
                        [score_name+': KO' for i in range(len(s_kos))] + \
                        ['Model: OK' for i in range(len(m_oks))] + \
                        ['Model: KO' for i in range(len(m_kos))]

                })
            colors = ['xkcd:cobalt', 'xkcd:cobalt', 'xkcd:bluish green', 'xkcd:bluish green']

            # effective plotting
            ax = axs[0][loader_n] 
            p = sb.kdeplot(
                    data = df,
                    ax = ax,
                    x = 'Value',
                    hue = 'Score',
                    common_norm = False,
                    palette = colors,
                    clip = [0., 1.],
                    alpha = 0.8,
                    legend = loader_n == 0
                    )

            lines = ['--', '-', '--', '-', '--', '-']
            # set up linestyles
            for ls, line in zip(lines, p.lines):
                line.set_linestyle(ls)
            
            # set legend linestyle
            if loader_n == 0:
                handles = p.legend_.legend_handles[::-1]
                for ls, h in zip(lines, handles):
                    h.set_ls(ls)

            ax.set_xlabel('Score')
            ax.set_ylabel('%')
            ax.title.set_text(f'{ds_key}\n{score_name} AUC={s_auc:.4f}\nModel AUC={m_auc:.4f}')
            ax.grid(True)

            # plot dropping-out accuracy
            _, s_idx = scores.sort() # scores is the protoscore
            _, m_idx = confs.sort() # confs is the max of the softmax
            s_acc = torch.zeros(drop_max+1)
            m_acc = torch.zeros(drop_max+1)
            for drop_perc in range(drop_max+1):
                n_drop = floor((drop_perc/100)*ns)
                s_acc[drop_perc] = 100*(results[s_idx[n_drop:]]).sum()/(ns-n_drop)
                m_acc[drop_perc] = 100*(results[m_idx[n_drop:]]).sum()/(ns-n_drop)
            
            colors = ['xkcd:cobalt', 'xkcd:bluish green']
            ax = axs[1][loader_n]
            df = pd.DataFrame({
                'Values': torch.hstack((s_acc, m_acc)),
                'Score': \
                        [score_name for i in range(drop_max+1)] + \
                        ['Model confidece' for i in range(drop_max+1)]
                })
            
            sb.lineplot(
                    data = df,
                    ax = ax,
                    x = torch.linspace(0, drop_max, drop_max+1).repeat(2),
                    y = 'Values',
                    hue = 'Score',
                    palette = colors,
                    alpha = 0.8,
                    legend = loader_n == 0,
                    )
            ax.set_xlabel('% dropped')
            ax.set_ylabel('Accuracy (%)')
            ax.grid(True)

            # plot callibration 
            n_bins = ceil(1/calib_bin)
            s_acc = torch.zeros(n_bins)
            m_acc = torch.zeros(n_bins)
            s_conf = torch.zeros(n_bins)
            m_conf = torch.zeros(n_bins)
            s_ns  = torch.zeros(n_bins)
            m_ns  = torch.zeros(n_bins)

            for b in range(n_bins):
                s_idx = torch.logical_and(scores > b*calib_bin, scores <= (b+1)*calib_bin)
                m_idx = torch.logical_and(confs > b*calib_bin, confs <= (b+1)*calib_bin)

                # how many samples in the bin
                s_ns[b] = s_idx.sum()
                m_ns[b] = m_idx.sum()

                # bin's accuracy
                s_acc[b] = (results[s_idx]).sum()/s_ns[b]
                m_acc[b] = (results[m_idx]).sum()/m_ns[b]

                # bin's average confidence
                s_conf[b] = scores[s_idx].sum()/s_ns[b]
                m_conf[b] = confs[m_idx].sum()/m_ns[b]

            # avoid numerical problems with NaN
            s_acc = torch.nan_to_num(s_acc) 
            m_acc = torch.nan_to_num(m_acc) 
            s_conf = torch.nan_to_num(s_conf) 
            m_conf = torch.nan_to_num(m_conf) 

            # Compute ECE score
            s_ece = (s_ns*(s_acc-s_conf).abs()).sum()/ns
            m_ece = (m_ns*(m_acc-m_conf).abs()).sum()/ns

            colors = ['xkcd:cobalt', 'xkcd:bluish green', 'xkcd:brick red']
            ax = axs[2][loader_n]
            df = pd.DataFrame({
                'Values': torch.hstack((s_acc*100, m_acc*100, torch.tensor([0, 1]))),
                'Score': \
                        [score_name for i in range(n_bins)] + \
                        [f'Model confidece' for i in range(n_bins)] + \
                        ['Perfect Calibration', 'Perfect Calibration']
                })
            
            sb.lineplot(
                    data = df,
                    ax = ax,
                    x = torch.hstack((torch.linspace(0, 1, n_bins).repeat(2), torch.tensor([0, 1]))),
                    y = 'Values',
                    hue = 'Score',
                    palette = colors,
                    alpha = 0.8,
                    legend = loader_n == 0,
                    )
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Accuracy (%)')
            ax.title.set_text(f'{ds_key}\n{score_name} ECE={s_ece:.2f}\nModel ECE={m_ece:.2f}')
            ax.grid(True)


    if plot:
        plt.savefig((path/phs.name).as_posix()+f'.{score_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    return ret 

def conceptogram_protoclass_score_attacks(**kwargs):
    phs_ori = kwargs.get('peepholes_ori')
    cvs_ori = kwargs.get('corevectors_ori')
    phs_atk = kwargs.get('peepholes_atk')
    cvs_atk = kwargs.get('corevectors_atk')
    loaders = kwargs.get('loaders', ['test'])
    target_modules = kwargs.get('target_modules', None)
    proto_key = kwargs.get('proto_key', 'train')
    bins = kwargs.get('bins', 20)
    plot = kwargs.get('plot', False)
    verbose = kwargs.get('verbose', False)
    atk_name = kwargs.get('atk_name', None)
    path = kwargs.get('path')

    # compute conceptogram entropy
    cpsso = phs_ori.get_conceptograms(loaders=loaders, target_modules=target_modules, verbose=verbose)
    cpssa = phs_atk.get_conceptograms(loaders=['test'], target_modules=target_modules, verbose=verbose)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        _bins = torch.arange(0, 1+1/bins, 1/bins)

    # sizes just to facilitate 
    nd = cpsso[loaders[0]].shape[1] # number of layers (distributions)
    nc = cpsso[loaders[0]].shape[2] # number of classes

    # compute proto-classes
    cps = cpsso[proto_key]
    results = cvs_ori._dss[proto_key]['result']
    labels = cvs_ori._dss[proto_key]['label']
    confs = sm(cvs_ori._dss[proto_key]['output'], dim=-1).max(dim=-1).values
    
    proto = torch.zeros(nc, nd, nc)
    for i in range(nc):
        cl = torch.logical_and(labels == i, results == 1)
        idx = torch.logical_and(cl, confs>0.9)
        _p = cps[idx].sum(dim=0)
        _p /= _p.sum(dim=1, keepdim=True)
        proto[i][:] = _p[:]

    # prepare returns dict
    ret = {
        'score': {},
        'auc': {},
        'protoclasses': proto,
        'so': {},
        'sa': {}
    }
    
    for loader_n, ds_key in enumerate(loaders):
        cps = cpsso[ds_key]
        ns = cps.shape[0] # number of samples
        results = cvs_ori._dss[ds_key]['result']
        labels = (cvs_ori._dss[ds_key]['label']).int()

        s = (proto[labels]*cps).sum(dim=(1,2))
        so = s/(torch.norm(proto[labels], dim=(1,2))*torch.norm(cps, dim=(1,2)))

    for loader_n, ds_key in enumerate(['test']):
        cps = cpssa[ds_key]
        ns = cps.shape[0] # number of samples
        results = cvs_atk._dss[ds_key]['result']
        labels = (cvs_atk._dss[ds_key]['label']).int()

        s = (proto[labels]*cps).sum(dim=(1,2))
        sa = s/(torch.norm(proto[labels], dim=(1,2))*torch.norm(cps, dim=(1,2)))

    idx = torch.argwhere((cvs_ori._dss[ds_key]['result'] == 1) & (cvs_atk._dss[ds_key]['attack_success'] == 1))

    so = so[idx].squeeze()
    sa = sa[idx].squeeze()

    # so = so.squeeze()
    # sa = sa.squeeze()
    
    lo = torch.ones(so.shape[0])
    la = torch.zeros(sa.shape[0])
    scores = torch.cat((so, sa))
    results = torch.cat((lo, la)) 

    try:
        auc_metric = AUC()
        auc_metric.update(scores, results.int())
        auc = auc_metric.compute().item()
    except Exception:
        auc = float('nan')

    ret['score'][ds_key] = scores 
    ret['auc'][ds_key] = auc
    ret['so'][ds_key] = so
    ret['sa'][ds_key] = sa

    #     if verbose: print(f'AUC for {ds_key} split: {auc:.4f}')
        
    # plotting
    if plot:
        ori = (so).detach().cpu().numpy()
        atk = (sa).detach().cpu().numpy()
        fpr, tpr, thresholds = roc_curve(results.detach().cpu().numpy(), scores.detach().cpu().numpy())
        plt.figure()
        fig, axs = plt.subplots(2,1, figsize=(7,10))
        axs[0].plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        axs[0].plot([0, 1], [0, 1], 'k--', label='Chance')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title('Receiver Operating Characteristic')
        axs[0].legend()
        axs[1].hist(ori, bins=50, label='ori')
        axs[1].hist(atk, bins=50, label=f'{atk_name}', alpha=0.7)
        axs[1].legend()
        # fig.savefig(f'../data/{name_model}/img/AUC/Feature_squeezing_attack={atk.replace("my", "")}_combined.png')
        
        
        # sb.histplot(data=pd.DataFrame({'score': ori}), ax=ax, bins=_bins, x='score', stat='density', label='ok n=%d'%len(ori), alpha=0.5)
        # sb.histplot(data=pd.DataFrame({'score': atk}), ax=ax, bins=_bins, x='score', stat='density', label='ko n=%d'%len(atk), alpha=0.5)
        # ax.set_xlabel('score: Proto-Class')
        # ax.set_ylabel('%')
        # ax.title.set_text(f'{ds_key} (AUC={auc:.4f})')
        # ax.legend(title='dist')

    if plot:
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
    return ret 


