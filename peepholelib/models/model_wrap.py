# General python stuff
from pathlib import Path as Path
import abc  

# torch stuff
import torch

class Hook:
    def __init__(self, save_input=True, save_output=False):
        self.module = None 
        self.handle = None

        self._si = save_input
        self._so = save_output
        
        self.in_shape = None
        self.out_shape = None

        self.in_activations = None 
        self.out_activations = None 
        return
    
    def register(self, module):
        # check is already registered to a module 
        if self.module or self.handle:
            self.handle.remove()
            self.handle = None
            self.module = None
        
        self.module = module 
        self.handle = module.register_forward_hook(self)
        return self.handle
    
    def set_shapes(self):
        if self._si:
            self.in_shape = self.in_activations.shape[1:]
        if self._so:
            self.out_shape = self.out_activations.shape[1:]

        return

    def __call__(self, module, module_in, module_out):
        if self._si: 
            self.in_activations = module_in[0]
        if self._so: 
            self.out_activations = module_out

        return

    def __str__(self):
        return f"\nInputs shape: {self.in_activations.shape}\nOutputs shape: {self.out_activations.shape}\n"

class ModelWrap(metaclass=abc.ABCMeta):

    from .svd import get_svds

    def __init__(self, **kwargs):
        # device for NN
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self._model = kwargs['model']

        # check and set model
        assert(issubclass(type(self._model), torch.nn.Module))
        self._model = self._model.to(self.device)
        self._model.eval()

        # set in set_model()
        self.num_classes = None

        # set in set_target_modules()
        self._target_modules = None 

        # computed in add_hooks()
        self._hooks = None
        self._si = None 
        self._so = None 
        
        # computed in get_svds()
        self._svds = None

        return

    def __call__(self, x):
        return self._model(x)
    
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

    def dry_run(self, **kwargs):
        '''
        A dry run is used to collect information from the module, such as activation's sizes

        Args:
        - x (tensor) - one input for the model set with set_model().
        '''
        _img = kwargs['x'].to(self.device)
        output = self._model(_img)
        self.num_classes = output.shape[1]
        
        if not self._hooks:
            raise RuntimeError('No hooks available. Please run set_hooks() first.')

        for hk in self._hooks:
            self._hooks[hk].set_shapes()

        return

    def get_target_modules(self):
        if not self._target_modules:
            raise RuntimeError('No target_modules available. Please run set_target_modules() first.')

        return self._target_modules

    def add_hooks(self, **kwargs):
        self._si = kwargs['save_input'] if 'save_input' in kwargs else True 
        self._so = kwargs['save_output'] if 'save_output' in kwargs else False 
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        
        if not self._target_modules:
            raise RuntimeError('No target_modules available. Please run set_target_modules() first.')

        _hooks = {}
        for key in self._target_modules:
            if verbose: print('Adding hook to module: ', key)

            module = self._target_modules[key]
            hook = Hook(save_input=self._si, save_output=self._so)
            handle = hook.register(module)

            _hooks[key] = hook
        
        self._hooks = _hooks
        return 

    def get_hooks(self):
        if not self._hooks:
            raise RuntimeError('No hooks available. Please run add_hooks() first.')
        return self._hooks
    
