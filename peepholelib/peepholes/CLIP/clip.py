# our stuff
from .classifier_base import DrillBase

# torch stuff
import torch
import clip

# python stuff
import abc 

class CLIP(DrillBase): 
    def __init__(self, **kwargs):
        DrillBase.__init__(self, **kwargs)

        self._model_name = kwargs['model_name'] if 'model_name' in kwargs else "ViT-B/32"
        self._model, self._preprocess = clip.load(self._model_name, device=self.device)
    
    @abc.abstractmethod
    def __call__(self, *args, **kwds):
       
       dss = kwds['dss']
       
       device = kwds['device']

       image_features = self._model.encode_image(images.to(device))

       mean_image = image_features.mean(dim=0, keepdim=True)           
       mean_image = mean_image / mean_image.norm(dim=-1, keepdim=True) 

       return mean_image
        
    @abc.abstractmethod
    def load(self, **kwargs):
       pass 

    @abc.abstractmethod
    def save(self, **kwargs):
       pass

    @abc.abstractmethod
    def fit(self, **kwargs):
       pass  
        
