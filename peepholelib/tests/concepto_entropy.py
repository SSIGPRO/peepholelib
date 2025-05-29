import torch
from torch.distributions import Categorical
from torch.nn.functional import softmax as sm


def ghl_score(cps, basis, weights):
    ns = cps.shape[0]
    nd = cps.shape[1]
    nc = cps.shape[2]
    
    # some checks
    if not weights == 'entropy' and not (isinstance(weights, list) and len(weights) == nd):
        raise RuntimeError('Weights should be \'entropy\' or a list with a value for each peephole in the conceptogram')
    
    if not basis == 'from-baricenter' and not (torch.is_tensor(basis) and basis.shape == torch.Size((ns, nc))):
        raise RuntimeError('Basis should be \'from-baricenter\' or a torch.tensor with shape == n_samples x n_peepholes in the conceptograms')


    if basis == 'from-baricenter':
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
    else:
        refs = basis

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
    return s

def ghl(cps, basis='from-baricenter', weights='entropy'):
    # to make it compliant with get_concemptograms()
    cps = cps.transpose(1, 2)
    # scores for min and max entropies
    s = ghl_score(cps, basis, weights)
    return 1-s

def gkl_score(cps):
    ns = cps.shape[0]
    nd = cps.shape[1]
    nc = cps.shape[2]
    bari = sm(cps.sum(dim=1)/nd, dim=1)

    refs = torch.zeros(ns, nc)
    bds = torch.inf*torch.ones(ns)
    for i in range(nc):
        base = torch.zeros(nc)
        base[i] = 1.0
        base = sm(base, dim=0)
        # KL distance form base
        bd = (base*((base/bari).log())).sum(dim=1)
        refs[bd<bds,:] = base
        bds[bd<bds] = bd[bd<bds]

    #(sm(b)*(sm(b)/sm(a)).log()).sum()
    refs = refs.unsqueeze(1)
    sm_cps = sm(cps, dim=2)
    dists = (refs*((refs/sm_cps).log())).sum(dim=2)
    
    ents = torch.zeros(ns, nd)
    for i in range(nd):
        ents[:,i] = Categorical(probs=cps[:,i,:]).entropy() 

    s = ((dists*ents).sum(dim=1))/ents.sum(dim=1)
    return s

def gkl(cps):
    # to make it compliant with get_concemptograms()
    cps = cps.transpose(1, 2)
    nd = cps.shape[1]
    nc = cps.shape[2]
    a = torch.zeros(nc)
    a[0] = 1 
    b = torch.zeros(nc)
    b[1] = 1 
    # scores for min and max entropies
    max_e = ((sm(a, dim=0)*(sm(a, dim=0)/sm(b, dim=0)).log()).sum())*((nd-1)/nd)
    min_e = 0.0
    s = gkl_score(cps)
    ns = 1-(s-min_e)/(max_e-min_e)
    return ns

if __name__ == "__main__":
    nd = 3
    nc = 4
    ns = 5

    rand = torch.rand((nc, nd)) 
    rand = rand/rand.sum(dim=0)

    line = torch.zeros((nc, nd))
    line[0, :] = 1.0
    
    half = torch.zeros(nc, nd)
    half[1, :] = 0.5
    half[2, :] = 0.5

    unif = (1/nc)*torch.ones((nc, nd))
    
    salt = torch.zeros((nc, nd))
    for j, i in enumerate(torch.randperm(nc)[:nd]):
        salt[i, j] = 1.
    
    #w = 'entropy'
    w = (torch.arange(nd)+1).tolist()
    #b = 'from-baricenter'
    b = torch.zeros(ns, nc)
    b[0,0] = b[1, 0] = b[2, 1] = b[3, 3] = b[4, salt.nonzero()[-1][0]] = 1.

    lbs = ['rand', 'line', 'half', 'unif', 'salt']
    cps = torch.stack([rand, line, half, unif, salt])
    for l, s in zip(lbs, ghl(cps, basis=b, weights=w)):
        print('-- score ', l, ': ', s)

