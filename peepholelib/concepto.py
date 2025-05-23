import torch

def entropy(x):
    return (-x*x.log()).sum()

def hist_entropy(x, bins=10):
    v, e = torch.histogram(x, bins=bins)
    print(v)
    return (-v*v.log()).sum()

if __name__ == "__main__":
    ds = 5
    rand = torch.rand((ds, ds)) 
    line = torch.zeros((ds, ds))
    line[torch.randint(0, ds, (1,)), :] = 1.0

    print('entropy rand: ', entropy(rand))
    print('entropy line: ', entropy(line))
    
    print('hist entropy rand: ', hist_entropy(rand))
    print('hist entropy line: ', hist_entropy(line))
