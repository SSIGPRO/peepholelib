from pathlib import Path as Path

import torch
import torch.nn as nn

class SubModel1(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        ds = kwargs['data_size']
        nf = kwargs['hidden_features']
        self.banana = nn.Sequential(
            nn.Linear(ds, nf),
            nn.Linear(nf, nf),
            nn.Linear(nf, ds)
        )
    
    # forward
    def forward(self, x):
        return self.banana(x)

class SubModel2(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.nn1 = SubModel1(**kwargs)
        self.nn2 = SubModel1(**kwargs)
    
    # forward
    def forward(self, x):
        return self.nn2(self.nn1(x))
    
class DummyModel(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.nn1 = SubModel2(**kwargs)
        self.nn2 = SubModel1(**kwargs)
        self.nn3 = SubModel1(**kwargs)
        
    # forward
    def forward(self, x):
        x1 = self.nn1(x)
        x2 = self.nn2(x)
        return self.nn3(x1+x2)
