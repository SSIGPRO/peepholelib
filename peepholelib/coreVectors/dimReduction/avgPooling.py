# torch stuff
import torch

# Our stuff
from .dim_reduction_base import DimReductionBase as DRB 

class AvgPooling(DRB):
    def __init__(self, **kwargs):
        DRB.__init__(self, **kwargs)
        
        # just to asset that the layer is Conv2d
        layer = kwargs['layer']
        model = kwargs['model']
        _layer = model._target_modules[layer]
        if not isinstance(_layer, torch.nn.Conv2d):
            raise RuntimeError("Only Conv2D is suported") 

        return

    def __call__(self, **kwargs):
        act_data = kwargs['act_data'] 
        act_data = act_data.view(act_data.size(0), act_data.size(1), -1)
        cvs = torch.mean(act_data, 2)
    
        return cvs

    def parser(self, **kwargs):
        cvs = kwargs['cvs']
        dss = kwargs.get('dss', None)
        label_key = kwargs.get('label_key', 'label') 

        ret = cvs if dss == None else (cvs, dss[label_key])
        return ret

