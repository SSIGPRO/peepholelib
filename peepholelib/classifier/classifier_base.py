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
    cvs = kwargs['cvs']
    act = kwargs['act'] if 'act' in kwargs else None
    layer = kwargs['layer']
    label_key = kwargs['label_key'] if 'label_key' in kwargs else 'label' 
    peep_size = kwargs['peep_size']

    if act == None:
        return cvs[layer][:,0:peep_size]
    else:
        return cvs[layer][:,0:peep_size], act[label_key]

def null_parser(**kwargs):
    data = kwargs['data']
    return data['data'], data['label'] 
    
class ClassifierBase: # quella buona
    def __init__(self, **kwargs):
        self.path = kwargs['path']
        self.name = kwargs['name']
        
        self.nl_class = kwargs['nl_classifier']
        self.nl_model = kwargs['nl_model']
        self.n_features = kwargs['n_features']

        self.parser = kwargs['parser'] if 'parser' in kwargs else null_parser 
        self.parser_kwargs = kwargs['parser_kwargs'] if 'parser_kwargs' in kwargs and 'parser' in kwargs else dict() 

        self.bs = kwargs['batch_size'] if 'batch_size' in kwargs else '64'
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        # computed in fit()
        self._classifier = None

        # set in fit()
        self._cvs = None 

        # computer in compute_empirical_posteriors()
        self._empp = None

        # defined in save() or load()
        self._empp_file = None
        self._clas_file = None
        self._suffix = f'{self.name}.n_features={self.n_features}.nl_class={self.nl_class}.nl_model={self.nl_model}'
        return
    
    @abc.abstractmethod
    def load(self, **kwargs):
        self._empp_file = self.path/(self._suffix+'.empp.pt')
        self._empp = torch.load(self._empp_file)
        pass 

    @abc.abstractmethod
    def save(self, **kwargs):
        self._empp_file = self.path/(self._suffix+'.empp.pt')
        torch.save(self._empp, self._empp_file)
        pass

    @abc.abstractmethod
    def fit(self, **kwargs):
        pass        

    @abc.abstractmethod
    def classifier_probabilities(self, **kwargs):
        pass
    
    def compute_empirical_posteriors(self, **kwargs):
        '''
        Compute the empirical posterior matrix P, where P(g, c) is the probability that a sample assigned to classifier's class g belongs to the model's class c.

        Args:
        - verbose (Bool): print some stuff
        '''
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        actds = kwargs['actds']
        corevds = kwargs['corevds']

        # pre-allocate empirical posteriors
        _empp = torch.zeros(self.nl_class, self.nl_model)
        
        # create dataloaders
        acts_dl = DataLoader(actds, batch_size=self.bs, collate_fn=lambda x: x)
        cvs_dl = DataLoader(corevds, batch_size=self.bs, collate_fn=lambda x: x)

        # iterate over _fit_data
        if verbose: print('Computing empirical posterior')
        for act, cvs in tqdm(zip(acts_dl, cvs_dl), disable=not verbose):
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
