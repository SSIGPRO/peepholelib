import torch
from torch.distributions import Categorical
from torch.nn.functional import softmax as sm

def cl_score(cps, weights=None):
    # to make it compliant with get_concemptograms()
    cps = cps.transpose(1, 2)
    
    ns = cps.shape[0]
    nd = cps.shape[1]
    nc = cps.shape[2]

    # for normalization
    _line = torch.zeros(nc)
    _line[0] = 1
    min_e = Categorical(probs=_line).entropy()
    _unif = torch.ones(nc)/nc
    max_e = Categorical(probs=_unif).entropy()

    # some checks
    if not weights == None and not (isinstance(weights, list) and len(weights) == nd):
        raise RuntimeError('Weights should be \'None\' or a list with a value for each peephole in the conceptogram')
    
    if weights == None:
        _w_cps = cps/nd
    else:
        _w = torch.tensor(weights).unsqueeze(1)
        _w_cps = cps*_w/_w.sum()

    # the values are already divided in the if statement above
    print('w cps: ', _w_cps)
    _means = _w_cps.sum(dim=1)
    _max = _means.max(dim=-1, keepdim=True).values
    _min = _means.min(dim=-1, keepdim=True).values
    print((_max-_min).squeeze())
    
    print('means: ', _means)
    s = Categorical(probs=_means).entropy() 

    return 1-(s-min_e)/(max_e-min_e)

def ghl_score(cps, basis='from_entropy', weights='entropy'):
    # to make it compliant with get_concemptograms()
    cps = cps.transpose(1, 2)

    ns = cps.shape[0]
    nd = cps.shape[1]
    nc = cps.shape[2]
    
    # some checks
    if not weights == 'entropy' and not (isinstance(weights, list) and len(weights) == nd):
        raise RuntimeError('Weights should be \'entropy\' or a list with a value for each peephole in the conceptogram')
    
    if not basis == 'from_baricenter' and not (torch.is_tensor(basis) and basis.shape == torch.Size((ns, nc))):
        raise RuntimeError('Basis should be \'from_baricenter\' or a torch.tensor with shape == n_samples x n_peepholes in the conceptograms')

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
    return 1-s

if __name__ == "__main__":
    nd = 3
    nc = 4
    ns = 5
    
    torch.manual_seed(32)

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
    w = torch.ones(nd).tolist()
    #b = 'from-baricenter'
    b = torch.zeros(ns, nc)
    b[0,0] = b[1, 0] = b[2, 1] = b[3, 3] = b[4, salt.nonzero()[-1][0]] = 1.

    lbs = ['rand', 'line', 'half', 'unif', 'salt']
    cps = torch.stack([rand, line, half, unif, salt])
    for l, s in zip(lbs, ghl_score(cps, basis=b, weights=w)):
        print('-- ghl score ', l, ': ', s)

    for l, s in zip(lbs, cl_score(cps, weights=w)):
        print('-- cl score ', l, ': ', s)
