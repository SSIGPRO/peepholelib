import torch
import abc 
from functools import partial
import sys
from pathlib import Path as Path

class HeadSqueezingDetector():
    from peepholelib.models.svd import get_svds
    from peepholelib.models.model_wrap import ModelWrap 

    def __init__(self, **kwargs):
        self._model = kwargs['model']
        self.score_fn = kwargs['score_fn'] if 'score_fn' in kwargs else partial(torch.norm, p=2, dim=1)
        
        self.output = None
        self.output_dict = {}

    def layer_SVD(self, **kwargs):
        path = Path(kwargs['path'])
        name = kwargs['name']
        self.target_module = kwargs['target_modules'] 
        sample_in = kwargs['sample_in']
        self.k = kwargs['rank'] if 'rank' in kwargs else 300
        channel_wise = kwargs['channel_wise'] if 'channel_wise' in kwargs else True
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False

        self._model.set_target_modules(target_modules=[self.target_module], verbose=verbose)

        self._model.get_svds(
                            path = path,
                            name = name,
                            target_modules = [self.target_module],
                            sample_in = sample_in,
                            rank = self.k,
                            channel_wise = channel_wise,
                            verbose = verbose
                            )

        self._model.set_activations(save_input=True, save_output=False)
    
    def __call__(self, **kwargs):
        image = kwargs['image']
        device = kwargs['device']
        mode = kwargs['mode'] if 'mode' in kwargs else 'first_k'
        n_iter = kwargs['n_iter'] if 'n_iter' in kwargs else 1000
        self.output_dict['ori'] = self._model(image.to(device))
        act_data = self._model._acts['in_activations'][self.target_module]
        n_act = act_data.shape[0]
        svd = self._model._svds[self.target_module]
        
        if mode == 'first_k':

            Vh = svd['Vh'][:self.k].T
            U = svd['U'][:self.k]
            s = svd['s'][:self.k]

            acts_flat = act_data.flatten(start_dim=1)
            ones = torch.ones(n_act, 1, device=device)
            _acts = torch.hstack((acts_flat, ones))
            
            A_ = Vh*s@U
            
            self.output_dict['squeezed'] = (A_.to(device).T@_acts.T).T
            self.output = self.score_fn(self.output_dict['ori'], self.output_dict['squeezed'])

        elif mode == 'random':

            score = []

            for i in range(n_iter):
                idx = torch.randperm(svd['Vh'].shape[0])[:self.k]
                
                Vh = svd['Vh'][idx].T
                U = svd['U'][idx]
                s = svd['s'][idx]

                acts_flat = act_data.flatten(start_dim=1)
                ones = torch.ones(n_act, 1, device=device)
                _acts = torch.hstack((acts_flat, ones))
                
                A_ = Vh*s@U
                
                self.output_dict['squeezed'] = (A_.to(device).T@_acts.T).T
                sfn = self.score_fn(self.output_dict['ori'] - self.output_dict['squeezed'])/ self.score_fn(self.output_dict['ori'])
                
                score.append(sfn)
            
            self.output = torch.stack(score).std(dim=0)
            
    
           
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'first_k' or 'random'.")

        
        return self.output

        


       