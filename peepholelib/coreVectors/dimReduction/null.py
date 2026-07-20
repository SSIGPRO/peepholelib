# Our stuff
from .dim_reduction_base import DimReductionBase as DRB 

class Null(DRB):
    def __init__(self, **kwargs):
        DRB.__init__(self, **kwargs)
        
        # just to asset that the layer is Conv2d
        layer = kwargs['layer']
        model = kwargs['model']

        return

    def __call__(self, **kwargs):
        """
        Simply flatten without any reduction. 
        """
        # Assuming the class token is the first token in the sequence

        act_data = kwargs['act_data']
        return act_data.flatten(start_dim=1)

    def parser(self, **kwargs):
        cvs = kwargs['cvs']
        dss = kwargs.get('dss', None)
        label_key = kwargs.get('label_key', 'label') 
                                                            
        ret = cvs if dss == None else (cvs, dss[label_key])
        return ret
