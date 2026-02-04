# General python stuff
from pathlib import Path as Path
import abc  

# torch stuff
import torch

import torch.nn as nn
from collections import OrderedDict
from torch import Tensor

'''Inspired by https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/architectures/utils_architectures.py'''

class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.normalizer = InputNormalizer(mean, std)
        self.model = model

    def forward(self, x):
        return self.model(self.normalizer(x))

class InputNormalizer(nn.Module):

    def __init__(self, mean, std):
        super(InputNormalizer, self).__init__()

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'InputNormalizer(mean={self.mean}, std={self.std})'  

def get_in_activations(x):
    return x['in_activations']

def get_out_activations(x):
    return x['out_activations']

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
    def __init__(self, **kwargs):
        # check and set model
        self._model = kwargs['model']
        assert(issubclass(type(self._model), torch.nn.Module))

        # impose requirse_grad = False for all parameters

        self.set_requires_grad(requires_grad=False, layer_names=None)

        # set target modules
        self._target_modules = None
        tm = kwargs.get('target_modules', None)
        if tm != None:
            self.set_target_modules(target_modules=tm) 

        # device for NN
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        # send model to device
        self._model = self._model.to(self.device)
        self._model.eval()

        # set in __call__()
        self._acts = None

        # set in set_activations()
        self._hooks = None
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
    
    def set_requires_grad(self, **kwargs):
        """
        Set requires_grad for model parameters.
        
        Args:
            requires_grad: Whether to enable gradients
            layer_names: Optional list of layer names to target. If None, affects all parameters.
        """
        layer_names = kwargs['layer_names']
        requires_grad = kwargs.get('requires_grad', True) 
        
        if layer_names is None:
            # Affect all parameters
            for param in self._model.parameters():
                param.requires_grad = requires_grad
        else:
            # Affect only specified layers
            for name, param in self._model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = requires_grad

    def _set_submodule(self, **kwargs):

        name = kwargs['name']
        new_mod = kwargs['new_mod']

        model = self._model

        if name is None:
            raise RuntimeError("No module name to replace.")
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = self._model.get_submodule(parent_name)
        else:
            parent, child_name = model, name
        parent._modules[child_name] = new_mod

    def update_output(self, **kwargs):
        n_classes = kwargs["to_n_classes"]
        output_layer = kwargs["output_layer"]  

        model = self._model
  
        layer = model.get_submodule(output_layer)
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"{output_layer} is not nn.Linear")
        new_layer = nn.Linear(layer.in_features, n_classes, bias=(layer.bias is not None), device=self.device)
        self._set_submodule(name=output_layer, new_mod=new_layer)
        return model
    

    def append_classifier(self, **kwargs):
        n_classes = kwargs["to_n_classes"]
        output_layer = kwargs["output_layer"]

        model = self._model

        layer = model.get_submodule(output_layer)
        if isinstance(layer, nn.Linear):
            out_size = layer.out_features
            new_head = nn.Linear(out_size, n_classes, bias=(layer.bias is not None), device=self.device)
            new_layer = nn.Sequential(layer, new_head)
            self._set_submodule(name=output_layer, new_mod=new_layer)
            return model

        if isinstance(layer, nn.Sequential) and len(layer) > 0 and isinstance(layer[-1], nn.Linear):
            out_size = layer[-1].out_features
            new_head = nn.Linear(out_size, n_classes, bias=(layer[-1].bias is not None), device=self.device)
            layer.append(new_head)
            return model

        raise TypeError(f"{output_layer} is not nn.Linear or nn.Sequential ending with nn.Linear")

    def load_checkpoint(self, **kwargs):
        '''
        Args:
        - name (str): name of file
        - path (pathlib.Path|str): folder path for file
        - sd_key (str): String for the model's state dict in the checkpoint. Defaults to `'state_dict'` 
        - verbose (bool): If True, print checkpoint information.
        
        Returns:
        - a thumbs up
        '''
        # kwargs
        _path = Path(kwargs['path'])
        _name = kwargs['name']
        sd_key = kwargs.get('sd_key', 'state_dict')
        verbose = kwargs.get('verbose', False)
        file = _path/_name
        
        # take the checkpoint and the state_dict from the saved file
        _checkpoint = torch.load(file, map_location=self.device)
        if sd_key in _checkpoint:
            _state_dict = _checkpoint[sd_key]
        else:
            _state_dict = _checkpoint  # Assume the entire model's state dictionary is stored directly
                    
        # assign model    
        self._model.load_state_dict(_state_dict) 
        
        return
    
    def normalize_model(self, **kwargs):
        '''
        Wrap the model with an InputNormalizer layer at the beginning.
        Args:
        - mean (torch.tensor): mean for each channel
        - std (torch.tensor): std for each channel
        '''

        mean = kwargs['mean']
        std = kwargs['std']

        mean = mean.to(self.device)
        std = std.to(self.device)
        
        layers = OrderedDict([('normalizer', InputNormalizer(mean, std)), ('model', self._model)])
        
        self._model = nn.Sequential(layers)

        return  
    
    def __get_module(self, **kwargs):
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
            _m = self.__get_module(key=_str)
            if _m != None:
                _dict[_str] = _m 

        self._target_modules = _dict
        
        return