# General pytho stuff
from math import ceil
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
        'protoclasses': proto,
        'so': {},
        'sa': {}
    }

    # for normalization
    _line = torch.zeros(nc)
    _line[0] = 1
    min_e = Categorical(probs=_line).entropy()
    _unif = torch.ones(nc)/nc
    max_e = Categorical(probs=_unif).entropy()
    
    for loader_n, ds_key in enumerate(loaders):
        cps = cpsso[ds_key]
        ns = cps.shape[0] # number of samples
        results = cvs_ori._dss[ds_key]['result']
        #labels = (cvs_ori._dss[ds_key]['label']).int()
        preds = cvs_ori._dss[proto_key]['pred'].int()

        s = (proto[preds]*cps).sum(dim=(1,2))
        so = s/(torch.norm(proto[preds], dim=(1,2))*torch.norm(cps, dim=(1,2)))

    for loader_n, ds_key in enumerate(['test']):
        cps = cpssa[ds_key]
        ns = cps.shape[0] # number of samples
        results = cvs_atk._dss[ds_key]['result']
        #labels = (cvs_atk._dss[ds_key]['label']).int()
        preds = cvs_ori._dss[proto_key]['pred'].int()

        s = (proto[preds]*cps).sum(dim=(1,2))
        sa = s/(torch.norm(proto[preds], dim=(1,2))*torch.norm(cps, dim=(1,2)))

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

