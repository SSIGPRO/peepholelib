# torch stuff
import torch

# Our stuff
from .dim_reduction_base import DimReductionBase as DRB 

class ViTTokenWiseMean(DRB):
    def __init__(self, **kwargs):
        DRB.__init__(self, **kwargs)
        
        # just to asset that the layer is Conv2d
        layer = kwargs['layer']
        model = kwargs['model']
        _layer = model._target_modules[layer]
        if not isinstance(_layer, torch.nn.Linear):
            raise RuntimeError("Only Linear is suported") 

        return

    def __call__(self, **kwargs):
        """
        Compute the channel-wise mean for ViT activations.
        """
        # Assuming the input is of shape (batch_size, num_tokens, num_channels)

        act_data = kwargs['act_data'] 
        act_data = act_data[:, 1:, :]  # Exclude the class token
        cvs = torch.mean(act_data, dim=2)  # Mean across tokens
        return cvs

    def parser(self, **kwargs):
        cvs = kwargs['cvs']
        dss = kwargs.get('dss', None)
        label_key = kwargs.get('label_key', 'label') 

        ret = cvs if dss == None else (cvs, dss[label_key])
        return ret
