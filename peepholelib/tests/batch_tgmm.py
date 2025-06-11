import torch
from torchgmm.bayes import GaussianMixture as tGMM
from torch.utils.data import DataLoader as DL


if __name__ == '__main__':
    dev = torch.device('cuda:0')
    ns = 50
    ds = 3
    bs = 10
    nc = 3 
    d = torch.rand(ns, ds)
    for i in range(ds):
        d[:, i] += i
        
    l = torch.randint(0, nc, (ns,))

    cl = tGMM(
            num_components = nc,
            batch_size = bs,
            trainer_params = dict(
                max_epochs = 1000,
                num_nodes = 1,
                accelerator = dev.type,
                devices = [dev.index]
                )
            )
    cl.fit(d)
    dt = torch.rand(ns, ds)
    p = cl.predict_proba(dt)
    print('p: ', p)

