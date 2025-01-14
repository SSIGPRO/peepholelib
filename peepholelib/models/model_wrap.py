# General python stuff
from pathlib import Path as Path
import abc  

# torch stuff
import torch

class Hook:
    def __init__(self, save_input=True, save_output=False):
        self.layer = None 
        self.handle = None

        self._si = save_input
        self._so = save_output
        
        self.in_shape = None
        self.out_shape = None

        self.in_activations = None 
        self.out_activations = None 
        return
    
    def register(self, layer):
        # check is already registered to a layer
        if self.layer or self.handle:
            self.handle.remove()
            self.handle = None
            self.layer = None
        
        self.layer = layer
        self.handle = layer.register_forward_hook(self)
        return self.handle
    
    def set_shapes(self):
        if self._si:
            self.in_shape = self.in_activations.shape[1:]
        if self._so:
            self.out_shape = self.out_activations.shape[1:]
        print('in out shapes: ', self.in_shape, self.out_shape)
        return

    def __call__(self, module, module_in, module_out):
        if self._si: 
            if self.in_activations == None or module_in[0].shape != self.in_activations.shape:
                self.in_activations = module_in[0]
            else:
                self.in_activations[:] = module_in[0][:]

        if self._so: 
            if self.out_activations == None or module_out.shape != self.out_activations.shape:
                self.out_activations = module_out
            else:
                self.out_activations[:] = module_out[:]
        return

    def __str__(self):
        return f"\nInputs shape: {self.in_activations.shape}\nOutputs shape: {self.out_activations.shape}\n"

class ModelWrap(metaclass=abc.ABCMeta):

    from .svd import get_svds

    def __init__(self, **kwargs):
        # device for NN
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        
        # set in set_model()
        self._model = None

        # set in set_target_layers()
        self._target_layers = None 

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


    def set_model(self, **kwargs):
        '''
        Set a nn as a model and apply the loaded checkpoint from a file.
        
        Args:
        - model (torch.nn): neural network
        - path (str|Path): Folder where the model is located.
        - name (str|Path): Model name.
        - verbose (bool): If True, print checkpoint information.
        
        Returns:
        - a thumbs up
        '''
        # kwargs
        model = kwargs['model']
        path = Path(kwargs['path'])
        name = Path(kwargs['name'])
        file = path/name
        
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        
        # take the checkpoint and the state_dict from the saved file
        self._checkpoint = torch.load(file, map_location=self.device)
        self._state_dict = self._checkpoint['state_dict']
        
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
        assert(issubclass(type(model), torch.nn.Module))
        
        self._model = model
        self._model.load_state_dict(self._state_dict) 
        self._model.to(self.device)
        self._model.eval()
        
        return

    def set_target_layers(self, **kwargs):
        '''
        Set the variable target_layers as a dictionary: the keys are the name of the layers (string) from the state_dict, the values are layers
   
        Args:
        - key_list (list): list of filtered keys from the state dict
        '''
        key_list = kwargs['target_layers']

        _dict = {}

        for _str in key_list:
            layer = self.get_layer(layername=_str)
            if layer != None:
                _dict[_str] = layer

        self._target_layers = _dict
        
        return

    def dry_run(self, **kwargs):
        '''
        A dry run is used to collect information from the module, such as activation's sizes

        Args:
        - x (tensor) - one input for the model set with set_model().
        '''
        _img = kwargs['x'].to(self.device)
        self._model(_img)    
        
        if not self._hooks:
            raise RuntimeError('No hooks available. Please run set_hooks() first.')

        for hk in self._hooks:
            self._hooks[hk].set_shapes()

        return

    def get_target_layers(self):
        if not self._target_layers:
            raise RuntimeError('No target_layers available. Please run set_target_layers() first.')

        return self._target_layers

    def add_hooks(self, **kwargs):
        self._si = kwargs['save_input'] if 'save_input' in kwargs else True 
        self._so = kwargs['save_output'] if 'save_output' in kwargs else False 
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        
        if not self._target_layers:
            raise RuntimeError('No target_layers available. Please run set_target_layers() first.')

        _hooks = {}
        for key in self._target_layers:
            if verbose: print('Adding hook to layer: ', key)

            layer = self._target_layers[key]
            hook = Hook(save_input=self._si, save_output=self._so)
            handle = hook.register(layer)

            _hooks[key] = hook
        
        self._hooks = _hooks
        return 

    def get_hooks(self):
        if not self._hooks:
            raise RuntimeError('No hooks available. Please run add_hooks() first.')
        return self._hooks
    
    def get_layer(self, **kwargs):
        '''
        Get the module of the neural network corresponding to the string passed as input
        
        Args:
        - layername (str): name of the layer we are searching for
        
        Returns:
        - temp: torch module
        '''
        temp = self._model
        layer_name = kwargs['layername']

        keys = layer_name.split(".")
        
        for p in keys:
            #check that all the strings in parts are actually keys of the dict temp._modules
            if p not in temp._modules.keys():
                return None
            temp = temp._modules[p]
            
        return temp
