import torch
import abc 
from functools import partial
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

class HeadSqueezingDetector(metaclass=abc.ABCMeta):
    from peepholelib.models.svds import get_svds

    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.score_fn = kwargs['score_fn'] if 'score_fn' in kwargs else partial(torch.norm, p=2, dim=1)
        
        self.output = None
        self.output_dict = {}

    def layer_SVD(self, **kwargs):
        path = Path(kwargs['path'])
        name = kwargs['name']
        self.target_module = kwargs['target_modules'] 
        sample_in = kwargs['sample_in']
        q = kwargs['rank'] if 'rank' in kwargs else 300
        channel_wise = kwargs['channel_wise'] if 'channel_wise' in kwargs else True
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False

        self.model.get_svds(
                            path = path,
                            name = name,
                            target_modules = [self.target_module],
                            sample_in = sample_in,
                            rank = q,
                            channel_wise = channel_wise,
                            verbose = verbose
                            )
        self.model.set_target_modules(target_modules=[self.target_module], verbose=verbose)

        self.model.set_activations(save_input=True, save_output=False)
    
    def __call__(self, image, device):
        self.output_dict['ori'] = self.model(image.to(device))
        act_data = self.model._acts['in_activations'][self.taget_module]
        n_act = act_data.shape[0]
        svd = self.model._svds[self.taget_module]

        acts_flat = act_data.flatten(start_dim=1)
        ones = torch.ones(n_act, 1, device=device)
        _acts = torch.hstack((acts_flat, ones))
        A_ = svd[self.taget_module]['U'].T@svd[self.taget_module]['s'].T@svd[self.taget_module]['Vh'].T
        self.output_dict['squeezed'] = (A_@_acts.T).T
        self. output = self.score_fn(self.output_dict['ori'] - self.output_dict['squeezed'])
        return self.output

        


       