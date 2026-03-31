import torch
from torch import nn as nn

class mySubNN(nn.Module):
    def __init__(self, size):
        nn.Module.__init__(self)

        self.nn = torch.nn.Sequential(
                nn.Linear(size, 4),
                nn.Linear(4, size)
                )

    def forward(self, x):
        return self.nn(x)
    
class myNN(nn.Module):
    def __init__(self, size):
        nn.Module.__init__(self)

        self.nn1 = mySubNN(size)
        self.nn2 = mySubNN(size)

    def forward(self, x):
        return self.nn2(self.nn1(x))
    

if __name__ == "__main__":
    ns = 3
    ds = 4
    ns = 2

    x = torch.rand(ns, ds)
    model = myNN(ds)

    pred = model(x)
    print('original pred: ', pred)
    
    last_module = model._modules['nn2']._modules['nn']
    out_features = last_module[-1].out_features
    print(out_features)
    last_module.append(nn.Linear(out_features, ns))

    pred = model(x)
    print('new pred: ', pred)

