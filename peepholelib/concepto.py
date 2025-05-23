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

def concepto_entropy(x):
    # torch accumulates on both not exactly what we want
    baricenter = x.sum(dim=1)/x.shape[1]
    dists = torch.zeros(x.shape)
    ents = torch.zeros(x.shape[1])
    for i in range(0, x.shape[1]):
        dists[:,i] = kl(x[:,i], baricenter, reduction='none') 
        ents[i] = Categorical(probs=x[:,i]).entropy()

    print(ents)
    print(dists)
    kull = (dists*ents).sum(dim=1)
    s = kull.sum()/sum(ents) 
    print('score: ', s)
    return

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
    
    sale = torch.zeros((cs, ls))
    for i in range(ls):
        sale[torch.randint(0, cs, (1,)), i] = 1.

    unif = (1/cs)*torch.ones((cs, ls))
    print('-- rand')
    concepto_entropy(rand)
    print('-- line')
    concepto_entropy(line)
    print('-- half')
    concepto_entropy(half)
    print('-- unif')
    concepto_entropy(unif)
    print('-- sale')
    concepto_entropy(sale)

    quit()

