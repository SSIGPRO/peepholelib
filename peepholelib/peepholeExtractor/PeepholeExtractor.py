# General python stuff
from pathlib import Path as Path
import abc  

# torch stuff
import torch

# our stuff
from peepholelib.coreVectors.get_coreVectors import get_in_activations
from matplotlib import pyplot as plt

class PeepholeExtractor(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        # device for NN
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.model = kwargs['model']
        
        self.target_layer = kwargs.get('target_layer')
        
        self.drillers = kwargs.get('dirllers')
        self.activations_parser = kwargs.get('activations_parser', get_in_activations)
        self.reduction_fn = kwargs.get('reduction_fn')
        self.norm_file = kwargs.get('norm_file')

        self.model.set_activations(save_input=True, save_output=True) 

        self.model._model.eval().zero_grad()

    def __call__(self, **kwargs):

        ds = kwargs.get('dss')
        ds_key = kwargs.get('ds_key', 'train')
        idxs = kwargs.get('idxs')

        samples = ds._dss[ds_key]['image'][idxs]
        plt.imshow(samples.permute(1,2,0).cpu())
        plt.savefig(f'sample_{idxs}.png')
        if not samples.requires_grad:
            samples = samples.requires_grad_(True)

        means, stds = torch.load(self.norm_file, weights_only=False)

        means, stds = means.to(self.device), stds.to(self.device)

        self.model(samples.to(self.device))

        a = self.activations_parser(self.model._acts)

        reduced = self.reduction_fn(act_data=a[self.target_layer])

        c = {self.target_layer: (reduced - means[self.target_layer]) / stds[self.target_layer]}

        n = self.drillers[self.target_layer].peephole_extract_grad_true(cvs=c) 
        
        return c, n, samples
