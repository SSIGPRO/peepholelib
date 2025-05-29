import torch
from torch.distributions import Categorical
from torch.nn.functional import softmax as sm


def ghl_score(x):
    ns = x.shape[0]
    nd = x.shape[1]
    nc = x.shape[2]
    
    rbari = (x.sum(dim=1)/nd).sqrt()
    #print('rbari: ', rbari)

    refs = torch.zeros(ns, nc)
    bds = torch.inf*torch.ones(ns)
    for i in range(nc):
        # note that base == base.sqrt()
        base = torch.zeros(nc)
        base[i] = 1.0
        #print('\n----- ', i)
        #print('base: ', base)
        # Hellinger distance form base
        bd = (torch.tensor(1/2).sqrt())*((rbari-base).norm(dim=1)) 
        #print('bd: ', bd)
        #print('bds:', bds)
        refs[bd<bds,:] = base
        bds[bd<bds] = bd[bd<bds]

    refs = refs.unsqueeze(1)
    #print('refs: ', refs)
    rx = x.sqrt()
    #print('rx: ', rx)
    dists = (torch.tensor(1/2).sqrt())*(rx-refs).norm(dim=2)
    #print('dists: ', dists)
    
    ents = torch.zeros(ns, nd)
    for i in range(nd):
        ents[:,i] = Categorical(probs=x[:,i,:]).entropy() 

    s = ((dists*ents).sum(dim=1))/ents.sum(dim=1)
    return s

def ghl(x):
    # to make it compliant with get_concemptograms()
    x = x.transpose(1, 2)
    # scores for min and max entropies
    s = ghl_score(x)
    return 1-s

def gkl_score(x):
    ns = x.shape[0]
    nd = x.shape[1]
    nc = x.shape[2]
    bari = sm(x.sum(dim=1)/nd, dim=1)

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
    sm_x = sm(x, dim=2)
    dists = (refs*((refs/sm_x).log())).sum(dim=2)
    print(dists)
    
    ents = torch.zeros(ns, nd)
    for i in range(nd):
        ents[:,i] = Categorical(probs=x[:,i,:]).entropy() 

    s = ((dists*ents).sum(dim=1))/ents.sum(dim=1)
    return s

def gkl(x):
    # to make it compliant with get_concemptograms()
    x = x.transpose(1, 2)
    nd = x.shape[1]
    nc = x.shape[2]
    a = torch.zeros(nc)
    a[0] = 1 
    b = torch.zeros(nc)
    b[1] = 1 
    # scores for min and max entropies
    max_e = ((sm(a, dim=0)*(sm(a, dim=0)/sm(b, dim=0)).log()).sum())*((nd-1)/nd)
    min_e = 0.0
    s = gkl_score(x)
    ns = 1-(s-min_e)/(max_e-min_e)
    return ns

if __name__ == "__main__":
    ls = 3
    cs = 4
    bss = [5, 10, 50, 100]

    rand = torch.rand((cs, ls)) 
    rand = rand/rand.sum(dim=0)

    line = torch.zeros((cs, ls))
    line[0, :] = 1.0
    
    half = torch.zeros(cs, ls)
    half[1, :] = 0.5
    half[2, :] = 0.5
    
    salt = torch.zeros((cs, ls))
    for j, i in enumerate(torch.randperm(cs)[:ls]):
        salt[i, j] = 1.
    
    unif = (1/cs)*torch.ones((cs, ls))
    
    lbs = ['rand', 'line', 'half', 'unif', 'salt']
    cps = torch.stack([rand, line, half, unif, salt])
    for l, s in zip(lbs, ghl(cps)):
        print('-- score ', l, ': ', s)

