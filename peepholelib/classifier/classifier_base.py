# python stuff
import abc  
from pathlib import Path
from tqdm import tqdm

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

def trim_corevectors(**kwargs):
    """
    Trims peephole data from a give layer.

    Args:
      tensor_dict (TensorDict): TensorDict from our CoreVectors class.
      layer (str): Layer key.

    Returns:
        nothing 
    """
    data = kwargs['data']
    layer = kwargs['layer']
    label = kwargs['label_key'] if 'label_key' in kwargs else 'label' 
    peep_size = kwargs['peep_size']
    return data['coreVectors'][layer][:,0:peep_size], data[label]

def null_parser(**kwargs):
    data = kwargs['data']
    return data['data'], data['label'] 
    
class ClassifierBase: # quella buona
    def __init__(self, **kwargs):

        load = kwargs['load'] if 'load' in kwargs else False

        if load:
            self.nl_class = None
            self.nl_model = None
            
            self.device = None
            self.parser = None
            self.parser_kwargs = {}
            
        else:
            self.nl_class = kwargs['nl_classifier']
            self.nl_model = kwargs['nl_model']
            
            self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
            self.parser = kwargs['parser'] if 'parser' in kwargs else null_parser 
            self.parser_kwargs = kwargs['parser_kwargs'] if 'parser_kwargs' in kwargs and 'parser' in kwargs else dict() 

        # set in fit()
        self._fit_dl = None

        # computed in fit()
        self._classifier = None

        # computer in compute_empirical_posteriors()
        self._empp = None

        return

    @abc.abstractmethod
    def fit(self, **kwargs):
        pass        

    @abc.abstractmethod
    def classifier_probabilities(self, **kwargs):
        pass

    def compute_empirical_posteriors(self, **kwargs):
        '''
        Compute the empirical posterior matrix P, where P(g, c) is the probability that a sample assigned to cluster g belongs to class c.

        Args:
        - verbose (Bool): print some stuff
        '''
        
        if self._fit_dl == None:
            raise RuntimeError('No fitting dataloader. Please run fit() first.')

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False

        # pre-allocate empirical posteriors
        _empp = torch.zeros(self.nl_class, self.nl_model)

        # iterate over _fit_data
        
        if verbose: print('Computing empirical posterior')
        for batch in tqdm(self._fit_dl, disable=not verbose):
            data, label = self.parser(data=batch, **self.parser_kwargs)
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
