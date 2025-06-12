import torch
import abc 
from functools import partial
import sys
from pathlib import Path as Path
from tqdm import tqdm

class HeadSqueezingDetector():
    from peepholelib.models.svd import get_svds
    from peepholelib.models.model_wrap import ModelWrap 

    def __init__(self, **kwargs):
        self._model = kwargs['model']
        self.device = kwargs['device']
        self.score_fn = kwargs['score_fn'] if 'score_fn' in kwargs else partial(torch.norm, p=2, dim=1)
        self.SM = torch.nn.Softmax(dim=1)
        
        self.output = None
        self.output_dict = {}

    def layer_SVD(self, **kwargs):
        path = Path(kwargs['path'])
        name = kwargs['name']
        self.target_module = kwargs['target_modules'] 
        sample_in = kwargs['sample_in']
        self.k = kwargs['rank'] if 'rank' in kwargs else 50
        channel_wise = kwargs['channel_wise'] if 'channel_wise' in kwargs else True
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False

        self._model.set_target_modules(target_modules=self.target_module, verbose=verbose)

        self._model.get_svds(
                            path = path,
                            name = name,
                            target_modules = self.target_module,
                            sample_in = sample_in,
                            rank = self.k,
                            channel_wise = channel_wise,
                            verbose = verbose
                            )

        self._model.set_activations(save_input=True, save_output=False)

        svd = self._model._svds[self.target_module[0]]

        self.Vh = svd['Vh'].to(self.device)
        self.U = svd['U'].to(self.device)
        self.s = svd['s'].to(self.device)
        
    def __call__(self, **kwargs):
        image = kwargs['image']
        # label = kwargs['label']
        mode = kwargs['mode'] if 'mode' in kwargs else 'first_k'
        n_iter = kwargs['n_iter'] if 'n_iter' in kwargs else 1000

        if mode == 'first_k':

            Vh = self.Vh[:self.k, :]
            U = self.U[:,:self.k]
            s = self.s[:self.k]
            S = torch.diag(s)
            self._A = U @ S @ Vh

            with torch.no_grad():
        
                self.output_dict['ori'] = self._model(image.to(self.device))

            act_data = self._model._acts['in_activations'][self.target_module[0]].to(self.device)
            n_act = act_data.shape[0]
            
            acts_flat = act_data.flatten(start_dim=1)
            ones = torch.ones(n_act, 1, device=self.device)
            _acts = torch.hstack((acts_flat, ones))
                
            self.output_dict['squeezed'] = (self._A@_acts.T).T                
            self.output = self.score_fn(self.output_dict['ori'] - self.output_dict['squeezed'])

        elif mode == 'random':
                    
                    score = []
                    with torch.no_grad():
        
                            self.output_dict['ori'] = self.SM(self._model(image.to(self.device)))
                    
                    for _ in tqdm(range(n_iter)):
                    
                        idx = torch.randperm(self.Vh.shape[0])[:self.k]

                        Vh = self.Vh[idx,:]
                        U = self.U[:,idx]
                        s = self.s[idx]
                        S = torch.diag(s)
                        self._A = U @ S @ Vh
                        
                        act_data = self._model._acts['in_activations'][self.target_module[0]].to(self.device)
                        n_act = act_data.shape[0]
                        
                        acts_flat = act_data.flatten(start_dim=1)
                        ones = torch.ones(n_act, 1, device=self.device)
                        _acts = torch.hstack((acts_flat, ones))
                            
                        self.output_dict['squeezed'] = self.SM((self._A@_acts.T).T)
                            
                        score.append(self.score_fn(self.output_dict['ori'] - self.output_dict['squeezed']))
                    
                    self.output = torch.var(torch.stack(score), dim=0)
                    # print(score.shape)

        elif mode == 'last_k':
            Vh = self.Vh[-self.k:, :]
            U = self.U[:,-self.k:]
            s = self.s[-self.k:]
            S = torch.diag(s)
            self._A = U @ S @ Vh

            with torch.no_grad():
        
                self.output_dict['ori'] = self.SM(self._model(image.to(self.device)))

            act_data = self._model._acts['in_activations'][self.target_module[0]].to(self.device)
            n_act = act_data.shape[0]
            
            acts_flat = act_data.flatten(start_dim=1)
            ones = torch.ones(n_act, 1, device=self.device)
            _acts = torch.hstack((acts_flat, ones))
                
            self.output_dict['squeezed'] = self.SM((self._A@_acts.T).T)
                
            self.output = self.score_fn(self.output_dict['ori'] - self.output_dict['squeezed'])
        else:

            raise ValueError(f"Unknown mode: {mode}. Use 'first_k', 'random', or 'last_k'.") 

        return self.output

        


       