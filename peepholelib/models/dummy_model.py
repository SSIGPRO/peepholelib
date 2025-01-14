from pathlib import Path as Path

import torch
import torch.nn as nn
import torch.nn.functional as F

class SubModel1(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        si = kwargs['input_size']
        output_size = kwargs['output_size']
        hidden_features = kwargs['hidden_features']
        self.banana = nn.Sequential(
            nn.Linear(si, hidden_features),
            nn.Linear(hidden_features, hidden_features),
            nn.Linear(hidden_features, output_size)
        )
    
    # forward
    def forwad(self, x):
        return self.nn1(x)

class SubModel2(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        si = kwargs['input_size']
        output_size = kwargs['output_size']
        hidden_features = kwargs['hidden_features']
        self.nn1 = SubModel1(**kwargs)
        self.nn2 = SubModel1(**kwargs)
    
    # forward
    def forwad(self, x):
        return self.nn2(self.nn1(x))
    
class DummyModel(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        si = kwargs['input_size']
        output_size = kwargs['output_size']
        hidden_features = kwargs['hidden_features']
        self.nn1 = SubModel2(**kwargs)
        self.nn2 = SubModel1(**kwargs)
        self.nn3 = SubModel1(**kwargs)
        
    # forward
    def forwad(self, x):
        x1 = self.nn1(x)
        x2 = self.nn2(x)
        return self.nn3(x1+x2)