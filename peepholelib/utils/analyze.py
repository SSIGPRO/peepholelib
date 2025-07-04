# General pytho stuff
from tqdm import tqdm
from math import floor

# plotting stuff
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd

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

def conceptogram_entropy_score(**kwargs):
    '''
    Compute the Entropy score of all conceptograms in `phs._phs[`loaders`]`. `target_modules` are passed to `ph.get_conceptograms()` so the evaluation only consider the indicated modules. The score is computed as the entropy (`torch.distributions.Categorical.entropy()`) of the histogram of the conceptograms into `bins` bins.
    If `plot=True` is passed, saves a KDE plot of the score and model confidence (`torch.nn.softmax(<model output>)`) for the correct and misclassified samples. `<model output>` is taken from `cvs`. 

    Args:
    - phs (peepholelib.peepholes.Peepholes): peepholes from which to take the conceptograms.
    - cvs (peepholelib.coreVectors.CoreVectors): corevectors respective to the `phs`.
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'].
    - target_modules (list[str]): list if target modules, as keys from the model `statedict`.
    - bins (int): number of bins to construct the histogram of conceptoframs. Weakly affects the score.
    - plot (bool): save figure with distributions of correctly and miss-classified samples.
    - verbose (bool): print progress messages.

    Returns:
    - ret (dict('score': dict(), 'auc':dict())): entropy scores and AUC for each values in `loaders`.  
    '''

    phs = kwargs.get('peepholes')
    cvs = kwargs.get('corevectors')
    loaders = kwargs.get('loaders', ['test'])
    target_modules = kwargs.get('target_modules', None)
    bins = kwargs.get('bins', 50)
    plot = kwargs.get('plot', False)
    verbose = kwargs.get('verbose', False)

    score_name = 'Entropy'

    # get conceptogram 
    cpss = phs.get_conceptograms(loaders=loaders, target_modules=target_modules, verbose=verbose)

    if plot:
        fig, axs = plt.subplots(2, len(loaders), sharex='row', sharey='row', figsize=(4*len(loaders), 4))

    # sizes just to facilitate 
    nd = cpss[loaders[0]].shape[1] # number of layers (distributions)
    nc = cpss[loaders[0]].shape[2] # number of classes

    # for normalization
    _line = torch.zeros(bins)
    _line[0] = 1
    min_e = Categorical(probs=_line).entropy()
    _unif = torch.ones(bins)/bins
    max_e = Categorical(probs=_unif).entropy()

    # prepare returns dict
    ret = {
        'score': {}, 
        'auc': {},
    }

    for loader_n, ds_key in enumerate(loaders):
        cps = cpss[ds_key]
        ns = cps.shape[0] # number of samples
        results = cvs._dss[ds_key]['result']
        confs = sm(cvs._dss[ds_key]['output'], dim=-1).max(dim=-1).values
        
        # main computation
        scores = torch.zeros(ns)
        for i in range(ns):
            v = torch.histogram(cps[i], bins).hist
            s = Categorical(probs=v).entropy()
            scores[i] = 1-(s-min_e)/(max_e-min_e)

        # compute AUC for score
        s_auc = AUC().update(scores, results.int()).compute().item()

        # Save returns
        ret['score'][ds_key] = scores 
        ret['auc'][ds_key] = s_auc

        if verbose: print(f'AUC for {ds_key} split: {s_auc:.4f}')
        
        # plotting
        if plot:
            s_oks = scores[results == True]
            s_kos = scores[results == False]
            m_oks = confs[results == True]
            m_kos = confs[results == False]

            # compute AUC for model 
            m_auc = AUC().update(confs, results.int()).compute().item()

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
                    legend = loader_n == 0,
                    )

            lines = ['--', '-', '--', '-']
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
            # plot dropping-out accuracy plot
            _, s_idx = scores.sort()
            _, m_idx = confs.sort()
            s_acc = torch.zeros(100)
            m_acc = torch.zeros(100)
            for drop_perc in range(100):
                n_drop = floor((drop_perc/100)*ns)
                s_acc[drop_perc] = 100*(results[s_idx[n_drop:]]).sum()/(ns-n_drop)
                m_acc[drop_perc] = 100*(results[m_idx[n_drop:]]).sum()/(ns-n_drop)
            
            colors = ['xkcd:cobalt', 'xkcd:bluish green']
            ax = axs[1][loader_n]
            df = pd.DataFrame({
                'Values': torch.hstack((s_acc, m_acc)),
                'Score': \
                        [score_name for i in range(100)] + \
                        ['Model confidece' for i in range(100)]
                })
                                                                                   
            sb.lineplot(
                    data = df,
                    ax = ax,
                    x = torch.linspace(0, 99, 100).repeat(2),
                    y = 'Values',
                    hue = 'Score',
                    palette = colors,
                    alpha = 0.8,
                    legend = loader_n == 0,
                    )
            ax.set_xlabel('% dropped')
            ax.set_ylabel('Accuracy (%)')
            
    if plot:
        plt.savefig((phs.path/phs.name).as_posix()+f'.{ds_key}.{score_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    return ret 

def conceptogram_protoclass_score(**kwargs):
    '''
    Compute the Proto-Class score of all conceptograms in `phs._phs[`loaders`]`. `target_modules` are passed to `ph.get_conceptograms()` so the evaluation only consider the indicated modules. The score is computed by comparing the conceptogram with the protoclasses. #TODO: Add paper or a full description.
    If `plot=True` is passed, saves a KDE plot of the score and model confidence (`torch.nn.softmax(<model output>)`) for the correct and misclassified samples. `<model output>` is taken from `cvs`. 

    Args:
    - phs (peepholelib.peepholes.Peepholes): peepholes from which to take the conceptograms.
    - cvs (peepholelib.coreVectors.CoreVectors): corevectors respective to the `phs`.
    - loaders (list[str]): loaders to consider, usually ['train', 'test', 'val'].
    - target_modules (list[str]): list if target modules, as keys from the model `statedict`.
    - proto_key (str): the key in `loaders` to get compute the protoclasses from.
    - plot (bool): save figure with distributions of correctly and miss-classified samples.
    - verbose (bool): print progress messages.

    Returns:
    - ret (dict('score': dict(), 'auc':dict(), 'protoclasses':torch.tensor)): entropy scores and AUC for each values in `loaders`, and `protoclasses` for all classes with `shape=[n_model, n_modules, n_model]`, where `n_model` is the number of labels and `n_modules` the number of modules in `target_modules`.  
    '''

    phs = kwargs.get('peepholes')
    cvs = kwargs.get('corevectors')
    loaders = kwargs.get('loaders', ['test'])
    target_modules = kwargs.get('target_modules', None)
    proto_key = kwargs.get('proto_key', 'train')
    plot = kwargs.get('plot', False)
    verbose = kwargs.get('verbose', False)
    
    score_name = 'Proto-Class'

    # get conceptogram 
    cpss = phs.get_conceptograms(loaders=loaders, target_modules=target_modules, verbose=verbose)

    if plot:
        fig, axs = plt.subplots(2, len(loaders), sharex='row', sharey='row', figsize=(4*len(loaders), 4))

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
        idx = torch.logical_and(cl, confs>0.99)
        _p = cps[idx].sum(dim=0)
        _p /= _p.sum(dim=1, keepdim=True)
        proto[i][:] = _p[:]

    # for normalization
    _line = torch.zeros(nc)
    _line[0] = 1
    min_e = Categorical(probs=_line).entropy()
    _unif = torch.ones(nc)/nc
    max_e = Categorical(probs=_unif).entropy()

    # prepare returns dict
    ret = {
        'score': {},
        'auc': {},
        'protoclasses': proto,
    }
    
    for loader_n, ds_key in enumerate(loaders):
        cps = cpss[ds_key]
        ns = cps.shape[0] # number of samples
        results = cvs._dss[ds_key]['result']
        labels = (cvs._dss[ds_key]['label']).int()
        confs = sm(cvs._dss[ds_key]['output'], dim=-1).max(dim=-1).values

        # main computation
        _wcps = (proto[labels]*cps).sum(dim=1)

        # normalization does not matter for entropy
        s = Categorical(probs=_wcps).entropy() 
        scores = 1-(s-min_e)/(max_e-min_e)

        # compute AUC for score
        s_auc = AUC().update(scores, results.int()).compute().item()

        ret['score'][ds_key] = scores 
        ret['auc'][ds_key] = s_auc

        if verbose: print(f'AUC for {ds_key} split: {s_auc:.4f}')
        
        # plotting
        if plot:
            s_oks = scores[results == True]
            s_kos = scores[results == False]
            m_oks = confs[results == True]
            m_kos = confs[results == False]
            
            # compute AUC for model 
            m_auc = AUC().update(confs, results.int()).compute().item()
            
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
                    legend = loader_n == 0,
                    )

            lines = ['--', '-', '--', '-']
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

            # plot dropping-out accuracy plot
            drop_max = 20
            _, s_idx = scores.sort()
            _, m_idx = confs.sort()
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

    if plot:
        plt.savefig((phs.path/phs.name).as_posix()+f'.{ds_key}.{score_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    return ret 
