# General pytho stuff
from tqdm import tqdm

# plotting stuff
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd

# torch stuff
import torch
from torch.distributions import Categorical
from torcheval.metrics import BinaryAUROC as AUC
from torch.nn.functional import  softmax as sm

def compute_top_k_accuracy(peepholes, targets, k):
    """
    peepholes: (Tensor [n_samples, n_classes])
    targets: (Tensor [n_samples]) - true class labels
    k: (int) - top-k to compute
    """
    topk = torch.topk(peepholes, k=k, dim=1).indices
    targets = targets.unsqueeze(1).expand_as(topk)
    correct = (topk == targets).any(dim=1).float()
    return correct.mean().item()

def conceptogram_cl_score(**kwargs):
    phs = kwargs.get('peepholes')
    cvs = kwargs.get('corevectors')
    loaders = kwargs.get('loaders', ['test'])
    weights = kwargs.get('weights', None) 
    bins = kwargs.get('bins', 20)
    target_modules = kwargs.get('target_modules', None)
    plot = kwargs.get('plot', False)
    verbose = kwargs.get('verbose', False)
    score_type = kwargs.get('score_type', 'entropy')

    # compute conceptogram entropy
    cpss = phs.get_conceptograms(loaders=loaders, target_modules=target_modules, verbose=verbose)

    if plot:
        fig, axs = plt.subplots(1, len(loaders), sharex='all', sharey='all', figsize=(4*len(loaders), 4))
        _bins = torch.arange(0, 1+1/bins, 1/bins)

    # sizes just to facilitate 
    nd = cpss[loaders[0]].shape[1] # number of layers (distributions)
    nc = cpss[loaders[0]].shape[2] # number of classes

    # for normalization
    _line = torch.zeros(nc)
    _line[0] = 1
    min_e = Categorical(probs=_line).entropy()
    _unif = torch.ones(nc)/nc
    max_e = Categorical(probs=_unif).entropy()

    # some checks
    if not weights == None and not (isinstance(weights, list) and len(weights) == nd):
        raise RuntimeError('Weights should be \'None\' or a list with a value for each peephole in the conceptogram')

    # prepare returns dict
    ret = {
        'score': {}, 
        'auc': {},
    }
    
    for loader_n, ds_key in enumerate(loaders):
        cps = cpss[ds_key]
        ns = cps.shape[0] # number of samples
        results = cvs._dss[ds_key]['result']
        
        if weights == None:
            _w_cps = cps/nd
        else:
            _w = torch.tensor(weights).unsqueeze(1)
            _w_cps = cps*_w/_w.sum()
                                                                   
        # the values are already divided in the if statement above
        _means = _w_cps.sum(dim=1)
        
        if score_type == 'entropy':
            s = Categorical(probs=_means).entropy() 
                                                                
            # normalize it
            scores = 1-(s-min_e)/(max_e-min_e)
        elif score_type == 'max_min':
            _max = _means.max(dim=-1, keepdim=True).values
            _min = _means.min(dim=-1, keepdim=True).values
            scores = (_max-_min).squeeze()

        try:
            auc_metric = AUC()
            auc_metric.update(scores, results.int())
            auc = auc_metric.compute().item()
        except Exception:
            auc = float('nan')
        
        # Save returns
        ret['score'][ds_key] = scores
        ret['auc'][ds_key] = auc

        if verbose: print(f'AUC for {ds_key} split: {auc:.4f}')

        # plotting
        if plot:
            oks = (scores[results == True]).detach().cpu().numpy()
            kos = (scores[results == False]).detach().cpu().numpy()

            ax = axs[loader_n] 
            sb.histplot(data=pd.DataFrame({'score': oks}), ax=ax, bins=_bins, x='score', stat='density', label='ok n=%d'%len(oks), alpha=0.5)
            sb.histplot(data=pd.DataFrame({'score': kos}), ax=ax, bins=_bins, x='score', stat='density', label='ko n=%d'%len(kos), alpha=0.5)
            ax.set_xlabel('score: CL')
            ax.set_ylabel('%')
            ax.title.set_text(f'{ds_key} (AUC={auc:.4f})')
            ax.legend(title='dist')

    if plot:
        plt.savefig((phs.path/phs.name).as_posix()+f'.{ds_key}.concepto_CL.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    return ret 

def conceptogram_entropy_score(**kwargs):
    phs = kwargs['peepholes']
    cvs = kwargs['corevectors']
    loaders = kwargs['loaders'] if 'loaders' in kwargs else ['test'] 
    bins = kwargs['bins'] if 'bins' in kwargs else 20 
    target_modules = kwargs.get('target_modules', None)
    plot = kwargs['plot'] if 'plot' in kwargs else False
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    # compute conceptogram entropy
    cpss = phs.get_conceptograms(loaders=loaders, target_modules=target_modules, verbose=verbose)

    if plot:
        fig, axs = plt.subplots(1, len(loaders), sharex='all', sharey='all', figsize=(4*len(loaders), 4))
        _bins = torch.arange(0, 1+1/bins, 1/bins)

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
        
        scores = torch.zeros(ns)
        for i in range(ns):
            v = torch.histogram(cps[i], bins).hist
            s = Categorical(probs=v).entropy()
            scores[i] = 1-(s-min_e)/(max_e-min_e)

        try:
            auc_metric = AUC()
            auc_metric.update(scores, results.int())
            auc = auc_metric.compute().item()
        except Exception:
            auc = float('nan')

        # Save returns
        ret['score'][ds_key] = scores 
        ret['auc'][ds_key] = auc

        if verbose: print(f'AUC for {ds_key} split: {auc:.4f}')
        
        # plotting
        if plot:
            oks = (scores[results == True]).detach().cpu().numpy()
            kos = (scores[results == False]).detach().cpu().numpy()

            ax = axs[loader_n] 
            sb.histplot(data=pd.DataFrame({'score': oks}), ax=ax, bins=_bins, x='score', stat='density', label='ok n=%d'%len(oks), alpha=0.5)
            sb.histplot(data=pd.DataFrame({'score': kos}), ax=ax, bins=_bins, x='score', stat='density', label='ko n=%d'%len(kos), alpha=0.5)
            ax.set_xlabel('score: Entropy')
            ax.set_ylabel('%')
            ax.title.set_text(f'{ds_key} (AUC={auc:.4f})')
            ax.legend(title='dist')

    if plot:
        plt.savefig((phs.path/phs.name).as_posix()+f'.{ds_key}.concepto_entropy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    return ret 

def conceptogram_protoclass_score(**kwargs):
    phs = kwargs.get('peepholes')
    cvs = kwargs.get('corevectors')
    loaders = kwargs.get('loaders', ['test'])
    target_modules = kwargs.get('target_modules', None)
    proto_key = kwargs.get('proto_key', 'train')
    bins = kwargs.get('bins', 20)
    plot = kwargs.get('plot', False)
    verbose = kwargs.get('verbose', False)

    # compute conceptogram entropy
    cpss = phs.get_conceptograms(loaders=loaders, target_modules=target_modules, verbose=verbose)

    if plot:
        fig, axs = plt.subplots(1, len(loaders), sharex='all', sharey='all', figsize=(4*len(loaders), 4))
        _bins = torch.arange(0, 1+1/bins, 1/bins)

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

        # main computation
        _wcps = (proto[labels]*cps).sum(dim=1)
        # normalization does not matter for entropy
        s = Categorical(probs=_wcps).entropy() 
        scores = 1-(s-min_e)/(max_e-min_e)

        try:
            auc_metric = AUC()
            auc_metric.update(scores, results.int())
            auc = auc_metric.compute().item()
        except Exception:
            auc = float('nan')


        ret['score'][ds_key] = scores 
        ret['auc'][ds_key] = auc

        if verbose: print(f'AUC for {ds_key} split: {auc:.4f}')
        
        # plotting
        if plot:
            oks = (scores[results == True]).detach().cpu().numpy()
            kos = (scores[results == False]).detach().cpu().numpy()
            
            ax = axs[loader_n] 
            sb.histplot(data=pd.DataFrame({'score': oks}), ax=ax, bins=_bins, x='score', stat='density', label='ok n=%d'%len(oks), alpha=0.5)
            sb.histplot(data=pd.DataFrame({'score': kos}), ax=ax, bins=_bins, x='score', stat='density', label='ko n=%d'%len(kos), alpha=0.5)
            ax.set_xlabel('score: Proto-Class')
            ax.set_ylabel('%')
            ax.title.set_text(f'{ds_key} (AUC={auc:.4f})')
            ax.legend(title='dist')

    if plot:
        plt.savefig((phs.path/phs.name).as_posix()+f'.{ds_key}.concepto_protoclass.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    return ret 
