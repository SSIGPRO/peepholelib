# python stuff
from functools import partial
import abc  

# torch stuff
import torch

class FeatureSqueezingDetector(metaclass=abc.ABCMeta):
    """
    Feature Squeezing Detector

    Args:
    - model (peepholelib.models.model_warp.ModelWrap): wrappped model.
    - prepro_dict (dict('str': peepholelib.featureSqueezing.preprocessing.XX )): Dictionary with keys being strings and values pre-processing modules implemented in `peepholelib.featureSqueezing`.
    """

    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.prepro_dict = kwargs['prepro_dict']
        self.output = None
        self.output_dict = {}
        self.score_dict = {}
        self.image_dict = {}
        self.calculate_score = kwargs['distance_function'] if 'distance_function' in kwargs else partial(torch.norm, p=1, dim=1)
        

    def __call__(self, image):
        """
        Detects feature squeezing in the given image.
        """
        # Implement detection logic here
        SM = torch.nn.Softmax(dim=1)
        output = self.model(image.to(self.model.device))
        self.output = SM(output)

        for key, filter in self.prepro_dict.items():
            filtered_image = filter(image)
            self.image_dict[key] = filtered_image
            output = self.model(filtered_image.to(self.model.device))

            self.output_dict[key] = SM(output)
            self.score_dict[key] = self.calculate_score(self.output - self.output_dict[key])
        
        self.score, _ = torch.max(torch.stack(list(self.score_dict.values())), dim=0)
    
        return self.score
