# Stuff used in evaluation ... will get out from here
from collections import Counter
import numpy as np

# plotting stuff
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd

# torch stuff
import torch
from torch.distributions import Categorical
from torcheval.metrics import BinaryAUROC as AUC
from torch.nn.functional import  softmax as sm

# TODO: give a better name
def evaluate_dists(**kwargs):
    phs = kwargs['peepholes']
    dss = kwargs['dataset']

    score_type = kwargs['score_type']
    bins = kwargs['bins'] if 'bins' in kwargs else 100
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False 

    for module in phs.target_modules:
        print(f'\n-------------\nEvaluating Distributions for module {module}\n-------------\n') 
        
        n_dss = len(phs._phs.keys())
        fig, axs = plt.subplots(1, n_dss+1, sharex='all', sharey='all', figsize=(4*(1+n_dss), 4))
        
        m_ok, s_ok, m_ko, s_ko = {}, {}, {}, {}

        for i, ds_key in enumerate(phs._phs.keys()):       # train val test
            if verbose: print(f'Evaluating {ds_key}')
            results = dss[ds_key]['result']
            scores = phs._phs[ds_key][module]['score_'+score_type]
            oks = (scores[results == True]).detach().cpu().numpy()
            kos = (scores[results == False]).detach().cpu().numpy()

            m_ok[ds_key], s_ok[ds_key] = oks.mean(), oks.std()
            m_ko[ds_key], s_ko[ds_key] = kos.mean(), kos.std()

            #--------------- 
            # plotting
            #---------------
            ax = axs[i+1]
            sb.histplot(data=pd.DataFrame({'score': oks}), ax=ax, bins=bins, x='score', stat='density', label='ok n=%d'%len(oks), alpha=0.5)
            sb.histplot(data=pd.DataFrame({'score': kos}), ax=ax, bins=bins, x='score', stat='density', label='ko n=%d'%len(kos), alpha=0.5)
            ax.set_xlabel('score: '+score_type)
            ax.set_ylabel('%')
            ax.title.set_text(ds_key)
            ax.legend(title='dist')
        
        # plot train and test distributions
        ax = axs[0]
        scores = phs._phs['train'][module]['score_'+score_type].detach().cpu().numpy()
        sb.histplot(data=pd.DataFrame({'score': scores}), ax=ax, bins=bins, x='score', stat='density', label='train n=%d'%len(scores), alpha=0.5)
        scores = phs._phs['val'][module]['score_'+score_type].detach().cpu().numpy()
        sb.histplot(data=pd.DataFrame({'score': scores}), ax=ax, bins=bins, x='score', stat='density', label='val n=%d'%len(scores), alpha=0.5)
        ax.set_ylabel('%')
        ax.set_xlabel('score: '+score_type)
        ax.legend(title='datasets')
        plt.savefig((phs.path/phs.name).as_posix()+'.'+score_type+f'.dists.{module}.png', dpi=300, bbox_inches='tight')
        plt.close()

        if verbose: print('oks mean, std, n: ', m_ok, s_ok, len(oks), '\nkos, mean, std, n', m_ko, s_ko, len(kos))

    return m_ok, s_ok, m_ko, s_ko

# TODO: give a better name
def evaluate(**kwargs): 
    phs = kwargs['peepholes']
    cvs = kwargs['corevectors']

    score_type = kwargs['score_type']
    
    for module in phs.target_modules:
        quantiles = torch.arange(0, 1, 0.001) # setting quantiles list
        prob_train = phs._phs['train'][module]['peepholes']
        prob_val = phs._phs['val'][module]['peepholes']
        
        # TODO: vectorize
        conf_t = phs._phs['train'][module]['score_'+score_type].detach().cpu() 
        conf_v = phs._phs['val'][module]['score_'+score_type].detach().cpu() 

        th = [] 
        lt = []
        lf = []

        c = cvs._actds['val']['result'].detach().cpu().numpy()
        cntt = Counter(c) 
        
        for q in quantiles:
            perc = torch.quantile(conf_t, q)
            th.append(perc)
            idx = torch.argwhere(conf_v > perc)[:,0]

            # TODO: vectorize
            cnt = Counter(c[idx]) 
            lt.append(cnt[True]/cntt[True]) 
            lf.append(cnt[False]/cntt[False])

        plt.figure()
        x = quantiles.numpy()
        y1 = np.array(lt)
        y2 = np.array(lf)
        plt.plot(x, y1, label='OK', c='b')
        plt.plot(x, y2, label='KO', c='r')
        plt.plot(np.array([0., 1.]), np.array([1., 0.]), c='k')
        plt.legend()
        plt.savefig((phs.path/phs.name).as_posix()+'.'+score_type+f'.conf.{module}.png', dpi=300, bbox_inches='tight')
        plt.close()

    return np.linalg.norm(y1-y2), np.linalg.norm(y1-y2)

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

def conceptogram_ghl_score(**kwargs):
    phs = kwargs['peepholes']
    cvs = kwargs['corevectors']
    loaders = kwargs['loaders'] if 'loaders' in kwargs else ['test'] 
    basis = kwargs['basis'] if 'basis' in kwargs else 'from_baricenter' 
    weights = kwargs['weights'] if 'weights' in kwargs else 'entropy' 
    bins = kwargs['bins'] if 'bins' in kwargs else 20 
    plot = kwargs['plot'] if 'plot' in kwargs else False
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    # compute conceptogram entropy
    cpss = phs.get_conceptograms(verbose=verbose, loaders=loaders)

    if plot:
        fig, axs = plt.subplots(1, len(loaders), sharex='all', sharey='all', figsize=(4*len(loaders), 4))
        _bins = torch.arange(0, 1+1/bins, 1/bins)

    # sizes just to facilitate 
    nd = cpss[loaders[0]].shape[1] # number of layers (distributions)
    nc = cpss[loaders[0]].shape[2] # number of classes

    # some checks
    if not weights == 'entropy' and not (isinstance(weights, list) and len(weights) == nd):
        raise RuntimeError(f'Weights should be \'entropy\' or a list with a value for each peephole in the conceptogram. Got {weights}')
    
    if not basis == 'from_baricenter' and not basis=='from_output':
        raise RuntimeError(f'Basis should be \'from_baricenter\' or \'from_output\'. Got {basis}')

    # for saving means and stds
    m_ok, s_ok, m_ko, s_ko = {}, {}, {}, {}
    for loader_n, ds_key in enumerate(loaders):
        cps = cpss[ds_key]
        ns = cps.shape[0] # number of samples
        
        if basis == 'from_baricenter':
            rbari = (cps.sum(dim=1)/nd).sqrt()
            refs = torch.zeros(ns, nc)
            bds = torch.inf*torch.ones(ns)
            for i in range(nc):
                # note that base == base.sqrt()
                base = torch.zeros(nc)
                base[i] = 1.0
                # Hellinger distance form base
                bd = (torch.tensor(1/2).sqrt())*((rbari-base).norm(dim=1)) 
                refs[bd<bds,:] = base
                bds[bd<bds] = bd[bd<bds]
        elif basis == 'from_output':
            _refs = torch.zeros(ns, nc)
            idx = cvs._dss[ds_key]['pred'].int()
            _refs[torch.arange(ns),idx] = 1.
            refs = _refs

        refs = refs.unsqueeze(1)
        rcps = cps.sqrt()
        dists = (torch.tensor(1/2).sqrt())*(rcps-refs).norm(dim=2)
        
        if weights == 'entropy':
            _w = torch.zeros(ns, nd)
            for i in range(nd):
                _w[:,i] = Categorical(probs=cps[:,i,:]).entropy() 
        else:
            _w = torch.tensor(weights).unsqueeze(0)

        s = ((dists*_w).sum(dim=1))/_w.sum(dim=1)
        # normalize it
        scores = 1-s

        # plotting 
        results = cvs._dss[ds_key]['result']
        oks = (scores[results == True]).detach().cpu().numpy()
        kos = (scores[results == False]).detach().cpu().numpy()
        m_ok[ds_key], s_ok[ds_key] = oks.mean(), oks.std()
        m_ko[ds_key], s_ko[ds_key] = kos.mean(), kos.std()

        # plotting
        if plot:
            ax = axs[loader_n] 
            sb.histplot(data=pd.DataFrame({'score': oks}), ax=ax, bins=_bins, x='score', stat='density', label='ok n=%d'%len(oks), alpha=0.5)
            sb.histplot(data=pd.DataFrame({'score': kos}), ax=ax, bins=_bins, x='score', stat='density', label='ko n=%d'%len(kos), alpha=0.5)
            ax.set_xlabel('score: Generalized f-KL')
            ax.set_ylabel('%')
            ax.title.set_text(ds_key)
            ax.legend(title='dist')

    if plot:
        plt.savefig((phs.path/phs.name).as_posix()+f'.{ds_key}.concepto_gHL.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    return scores, m_ok, s_ok, m_ko, s_ko

def conceptogram_cl_score(**kwargs):
    phs = kwargs['peepholes']
    cvs = kwargs['corevectors']
    loaders = kwargs['loaders'] if 'loaders' in kwargs else ['test'] 
    basis = kwargs['basis'] if 'basis' in kwargs else 'from_baricenter' 
    weights = kwargs['weights'] if 'weights' in kwargs else None 
    bins = kwargs['bins'] if 'bins' in kwargs else 20 
    plot = kwargs['plot'] if 'plot' in kwargs else False
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    # compute conceptogram entropy
    cpss = phs.get_conceptograms(verbose=verbose, loaders=loaders)

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

    # prepare metrics dict
    metrics = {
        'auc': {},
        'm_ok': {},
        's_ok': {},
        'm_ko': {},
        's_ko': {}
    }
    
    for loader_n, ds_key in enumerate(loaders):
        cps = cpss[ds_key]
        ns = cps.shape[0] # number of samples
        
        if weights == None:
            _w_cps = cps/nd
        else:
            _w = torch.tensor(weights).unsqueeze(1)
            _w_cps = cps*_w/_w.sum()
                                                                   
        # the values are already divided in the if statement above
        _means = _w_cps.sum(dim=1)
        
        s = Categorical(probs=_means).entropy() 
                                                                   
        # normalize it
        scores = 1-(s-min_e)/(max_e-min_e)

        # plotting 
        results = cvs._dss[ds_key]['result']
        oks = (scores[results == True]).detach().cpu().numpy()
        kos = (scores[results == False]).detach().cpu().numpy()

        metrics['m_ok'][ds_key] = oks.mean()
        metrics['s_ok'][ds_key] = oks.std()
        metrics['m_ko'][ds_key] = kos.mean()
        metrics['s_ko'][ds_key] = kos.std()

        try:
            auc_metric = AUC()
            auc_metric.update(scores.detach().cpu(), results.to(dtype=torch.int32).detach().cpu())
            auc = auc_metric.compute().item()
        except Exception:
            auc = float('nan')
        metrics['auc'][ds_key] = auc

        if verbose:
            print(f'AUC for {ds_key} split: {auc:.4f}')
        
        # plotting
        if plot:
            ax = axs[loader_n] 
            sb.histplot(data=pd.DataFrame({'score': oks}), ax=ax, bins=_bins, x='score', stat='density', label='ok n=%d'%len(oks), alpha=0.5)
            sb.histplot(data=pd.DataFrame({'score': kos}), ax=ax, bins=_bins, x='score', stat='density', label='ko n=%d'%len(kos), alpha=0.5)
            ax.set_xlabel('score: Generalized f-KL')
            ax.set_ylabel('%')
            ax.title.set_text(f'{ds_key} (AUC={auc:.4f})')
            ax.legend(title='dist')

    if plot:
        plt.savefig((phs.path/phs.name).as_posix()+f'.{ds_key}.concepto_CL.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    return scores, metrics
