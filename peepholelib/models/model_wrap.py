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
        self._path = Path(kwargs['path'])
        self._name = kwargs['name']

        # check and set model
        assert(issubclass(type(self._model), torch.nn.Module))
        self._model = self._model.to(self.device)
        self._model.eval()

        # set in set_model()
        self.num_classes = None

        # set in set_target_modules()
        self._target_modules = None 

        # computed in load_checkpoint()
        self._checkpoint = None
        self._state_dict = None
        
        # computed in add_hooks()
        self._hooks = None
        self._si = None 
        self._so = None 
        
        # computed in get_svds()
        self._svds = None

        return

    def __call__(self, x):
        return self._model(x)

    def load_checkpoint(self, **kwargs):
        '''
        Args:
        - verbose (bool): If True, print checkpoint information.
        
        Returns:
        - a thumbs up
        '''
        # kwargs

        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        file = self._path/self._name
        
        # take the checkpoint and the state_dict from the saved file
        self._checkpoint = torch.load(file, map_location=self.device)
        if 'state_dict' in self._checkpoint:
            self._state_dict = self._checkpoint['state_dict']
        else:
            self._state_dict = self._checkpoint  # Assume the entire model's state dictionary is stored directly
                    
        # verbose - see what is saved in the checkpoint (except for the state_dict)
        if verbose:
            print('\n-----------------\ncheckpoint\n-----------------')
            for k, v in self._checkpoint.items():
                if k != 'state_dict':
                    print(k, v)
                else:
                    print('state_dict keys: \n', v.keys(), '\n')
            print('-----------------\n')
        
        # assign model    
        self._model.load_state_dict(self._state_dict) 
        
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
            #check that all the strings in parts are actually keys of the dict temp._modules
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
    
