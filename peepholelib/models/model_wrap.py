# General python stuff
from pathlib import Path as Path
import abc  

# torch stuff
import torch

import torch.nn as nn
from collections import OrderedDict
from torch import Tensor

means = {
        'CIFAR10': torch.tensor([0.424, 0.415, 0.384]),
        'CIFAR100': torch.tensor([0.438, 0.418, 0.377]),
        'ImageNet': torch.tensor([0.485, 0.456, 0.406]),
        'SVHN': torch.tensor([0.438, 0.444, 0.473])
        }

stds = {
        'CIFAR10': torch.tensor([0.283, 0.278, 0.284]),
        'CIFAR100': torch.tensor([0.300, 0.287, 0.294]),
        'ImageNet': torch.tensor([0.229, 0.224, 0.225]),
        'SVHN': torch.tensor([0.198, 0.201, 0.197]),
        }

'''Inspired by https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/architectures/utils_architectures.py'''

class ImageNormalizer(nn.Module):

    def __init__(self, mean, std):
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', mean.view(1, 3, 1, 1))
        self.register_buffer('std', std.view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'  

class Hook:
    def __init__(self, save_input=True, save_output=False):
        self.module = None 
        self.handle = None

        self._si = save_input
        self._so = save_output
        
        self.i_act = None 
        self.o_act = None 
        return
    
    def register(self, module):
        # check is already registered to a module 
        if self.module or self.handle:
            self.unregister()        
        
        self.module = module 
        self.handle = module.register_forward_hook(self)
        return self.handle
    
    def unregister(self):
        if self.handle:
            self.handle.remove()
        self.handle = None
        self.module = None
        return

    def __call__(self, module, module_in, module_out):
        if self._si: 
            self.i_act = module_in[0]
        if self._so: 
            self.o_act = module_out

        return

    def __str__(self):
        return f"\nInputs shape: {self.i_act.shape}\nOutputs shape: {self.o_act.shape}\n"

class ModelWrap(metaclass=abc.ABCMeta):

    from .get_svds import get_svds

    def __init__(self, **kwargs):
        # device for NN
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self._model = kwargs['model']

        # check and set model
        assert(issubclass(type(self._model), torch.nn.Module))
        self._model = self._model.to(self.device)
        self._model.eval()

        # set in set_target_modules()
        self._target_modules = None 
        self._hooks = None
        
        # computed in get_svds()
        self._svds = None

        # set on __call__()
        self._acts = None

        # set on set_activation()
        self._si = False 
        self._so = False 

        return
    
    def set_activations(self, **kwargs):
        '''
        Set the model to save activations upon __call__()

        Args:
        - save_input (bool): True to save IN activations, False ignores activations
        - save_output (bool): True to save OUT activations, False ignores activations
        - verbose (bool): print progress messages
        '''
        # Hooks params
        self._si = kwargs['save_input'] if 'save_input' in kwargs else False 
        self._so = kwargs['save_output'] if 'save_output' in kwargs else False 
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False

        if (not self._si) and (not self._so):
            if self._hooks: 
                if verbose: print('Not saving activations. Removing Hooks')

                for key in self._hooks:
                    self._hooks[key].unregister()

                self._hooks = None
                self._acts = None
            return
        else:
            _hooks = {}
            for key in self._target_modules:
                if verbose: print('Adding hook to module: ', key)
                                                                       
                module = self._target_modules[key]
                hook = Hook(save_input=self._si, save_output=self._so)
                handle = hook.register(module)
                                                                       
                _hooks[key] = hook
            
            self._hooks = _hooks

        return

    def __call__(self, x):
        '''
        Forwards the input through the model, and save activations if they are setted (see 'set_activations()') in self._acts.
        
        Args:
            x (torch.tensor): the input
        Returns:
            res (torch.tensor): the model output
        '''
        res = self._model(x)

        # get activations in a dict (similar to corevectors structure)
        if self._si or self._so:
            self._acts = {}
            if self._si: self._acts['in_activations'] = {}
            if self._so: self._acts['out_activations'] = {}

            for mk in self._target_modules:
                if self._si:
                    self._acts['in_activations'][mk] = self._hooks[mk].i_act
                
                if self._so:
                    self._acts['out_activations'][mk] = self._hooks[mk].o_act 

        return res 
    
    def update_output(self, **kwargs):
        '''
        Update the model output to one with size `to_n_classes`. This is done by substituting the model's output by, or appending a, new `torch.nn.Linear` layer accoding to `overwrite`.
        The last layer is assumed to be `torch.nn.Linear` within a `torch.nn.Sequential` module.

        Args:
        - output_layer (str): The output layer of the model as per the state dict key
        - to_n_classes (int): Number of output features of the new output layers 
        - overwrite (bool): If True, the last layer will be substituted by a new (randomly initialize) layer with the same `in_features` and `out_features = to_n_classes`. If False a new `torch.nn.Linear(out_features, to_n_classes)` layer will be appended after the `output_layer` layer. 
        
        Returns:
        - a thumbs up
        '''

        out_layer = kwargs['output_layer']
        n_classes = kwargs['to_n_classes']
        overwrite = kwargs['overwrite'] if  'overwrite' in kwargs else False
        
        keys = out_layer.split(".")[:-1]
        temp = self._model
        for p in keys:
            #check that string part is actually a key in temp._modules
            if p not in temp._modules.keys():
                raise RuntimeError(f'seems like {p} is not in the NN, are you sure the output_layer is correct?') 
            temp = temp._modules[p]

        if not isinstance(temp, torch.nn.Sequential):
            raise RuntimeError('Last module should be torch.nn.Sequential(). If you update the logic to handle any type of network, please submitt a PR.')

        if not isinstance(temp[-1], torch.nn.Linear):
            raise RuntimeError('Last layer is not a linear layer. I will not change it.')
        
        if overwrite:
            in_size = temp[-1].in_features
            temp[-1] = torch.nn.Linear(in_size, n_classes, device=self.device)
        else:
            out_size = temp[-1].out_features 
            temp.append(torch.nn.Linear(out_size, n_classes, device=self.device))

        return

    def load_checkpoint(self, **kwargs):
        '''
        Args:
        - name (str): name of file
        - path (pathlib.Path|str): folder path for file
        - verbose (bool): If True, print checkpoint information.
        
        Returns:
        - a thumbs up
        '''
        # kwargs
        _path = Path(kwargs['path'])
        _name = kwargs['name']
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        file = _path/_name
        
        # take the checkpoint and the state_dict from the saved file
        _checkpoint = torch.load(file, map_location=self.device)
        if 'state_dict' in _checkpoint:
            _state_dict = _checkpoint['state_dict']
        else:
            _state_dict = _checkpoint  # Assume the entire model's state dictionary is stored directly
                    
        # assign model    
        self._model.load_state_dict(_state_dict) 
        
        return
    
    def normalize_model(self, **kwargs):
        '''
        Wrap the model with an ImageNormalizer layer at the beginning.
        Args:
        - mean (torch.tensor): mean for each channel
        - std (torch.tensor): std for each channel
        '''

        dss = kwargs['dataset']

        mean = means[dss].to(self.device)
        std = stds[dss].to(self.device)
        layers = OrderedDict([('normalize', ImageNormalizer(mean, std)),
                            ('model', self._model)])
        
        self._model = nn.Sequential(layers)

        return 
    
    def get_module(self, **kwargs):
        '''
        Get the module of the neural network corresponding to the string passed as input
        
        Args:
        - key (str): name of the module we are searching for
        
        Returns:
        - temp: torch module
        '''
        temp = self._model
        module_name = kwargs['key']
        keys = module_name.split(".")

        for p in keys:
            #check that string part is actually a key in temp._modules
            if p not in temp._modules.keys():
                return None
            temp = temp._modules[p]
            
        return temp

    def set_target_modules(self, **kwargs):
        '''
        Set the variable target_modules as a dictionary: the keys are the name of the modules (string) from the state_dict, the values are modules 
   
        Args:
        - target_modules (list): list of keys from the state dict
        '''
        key_list = kwargs['target_modules']
        
        _dict = {}
        for _str in key_list:
            _m = self.get_module(key=_str)
            if _m != None:
                _dict[_str] = _m 

        self._target_modules = _dict
        
        return

