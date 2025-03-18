from tqdm import tqdm

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

def compute_empirical_posteriors( **kwargs):
        '''
        Compute the empirical posterior matrix P, where P(g, c) is the probability that a sample assigned to cluster g belongs to class c.

        Args:
        - verbose (Bool): print some stuff
        '''
        if self._cvs_dl == None:
            raise RuntimeError('No fitting dataloader. Please run fit() first.')

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        # pre-allocate empirical posteriors
        _empp = torch.zeros(self.nl_class, self.nl_model)

        # iterate over _fit_data
        
        if verbose: print('Computing empirical posterior')
        for act, cvs in tqdm(zip(self._act_dl, self._cvs_dl), disable=not verbose):
            data, label = self.parser(act=act, cvs=cvs, **self.parser_kwargs)
            data, label = data.to(self.device), label.to(self.device)
            preds = self._classifier.predict(data)
            
            for p, l in zip(preds, label):
                _empp[int(p), int(l)] += 1
       
        # normalize to get empirical posteriors
        _empp /= _empp.sum(dim=1, keepdim=True)

        # replace NaN with 0
        _empp = torch.nan_to_num(_empp)
        self._empp = _empp
        
        return 