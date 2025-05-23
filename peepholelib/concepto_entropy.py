import torch
from torch.distributions import Categorical
from torch.nn.functional import kl_div as kl

def entropy(x):
    return (-x*x.log2()).sum()

def hist_entropy(x, bins=10):
    v, e = torch.histogram(x, bins=bins)
    v = v[v>0]
    return (1/x.numel())*((-v*v.log2()).sum())

def cat_entropy(x, bins=10):
    v, e = torch.histogram(x, bins=bins)
    return Categorical(probs=v/x.numel()).entropy()

def gf_score(x):
    ns = x.shape[0]
    nc = x.shape[1]
    nd = x.shape[2]
    baris = x.sum(dim=2)/nd
    dists = torch.zeros(x.shape)
    ents = torch.zeros(ns, nd)

    for i in range(nd):
        dists[:,:,i] = kl(x[:,:,i], baris, reduction='none') 
        ent = Categorical(probs=x[:,:,i]).entropy()
        ents[:,i] = ent 
    ents = ents.reshape(ns, 1, nd)

    kull = (dists*ents).sum(dim=2)
    s = kull.sum(dim=1)/ents.sum(dim=2).squeeze() 
    return s

def concepto_entropy(x):
    # scores for min and max entropies
    u = torch.ones(x.shape[1])/x.shape[1]
    max_e = u*(u.log()-u)*x.shape[1]
    min_e = -1.0
    s = gf_score(x)
    ns = 1-(s-min_e)/(max_e-min_e)
    return ns

if __name__ == "__main__":
    ls = 3
    cs = 5
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
    for l, s in zip(lbs, concepto_entropy(cps)):
        print('-- score ', l, ': ', s)

