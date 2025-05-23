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
from torch.nn.functional import kl_div as kl

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

def conceptogram_entropy(**kwargs):
    phs = kwargs['peepholes']
    cvs = kwargs['corevectors']
    loaders = kwargs['loaders'] if 'loaders' in kwargs else ['test'] 
    bins = kwargs['bins'] if 'bins' in kwargs else 20 
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    plot = kwargs['plot'] if 'plot' in kwargs else False

    # compute conceptogram entropy
    cpss = phs.get_conceptograms(verbose=verbose, loaders=loaders)

    if plot:
        fig, axs = plt.subplots(1, len(loaders), sharex='all', sharey='all', figsize=(4*len(loaders), 4))
        _bins = torch.arange(0, 1+1/bins, 1/bins)

    # for saving means and stds
    m_ok, s_ok, m_ko, s_ko = {}, {}, {}, {}
    for loader_n, ds_key in enumerate(loaders):
        cps = cpss[ds_key].transpose(1, 2)

        # get min and max scores for normalization
        u = torch.tensor(1/cps.shape[1])
        max_e = u*(u.log()-u)*cps.shape[1]
        min_e = -1.0
        # sizes just to facilitate 
        ns = cps.shape[0] # number of samples
        nc = cps.shape[1] # number of classes
        nd = cps.shape[2] # number of layers (distributions)

        # Main computations
        baris = cps.sum(dim=2)/nd
        dists = torch.zeros(cps.shape)
        ents = torch.zeros(ns, nd)
        
        # vectorized along the whole dataset, iterate over layers
        for i in range(nd):
            dists[:,:,i] = kl(cps[:,:,i], baris, reduction='none') 
            ent = Categorical(probs=cps[:,:,i]).entropy()
            ents[:,i] = ent 
        ents = ents.reshape(ns, 1, nd)
                                                                  
        kull = (dists*ents).sum(dim=2)
        s = kull.sum(dim=1)/ents.sum(dim=2).squeeze() 
        
        # normalize it
        scores = 1-(s-min_e)/(max_e-min_e)
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
        plt.savefig((phs.path/phs.name).as_posix()+f'.{ds_key}.concepto_gfKL.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    return scores, m_ok, s_ok, m_ko, s_ko

