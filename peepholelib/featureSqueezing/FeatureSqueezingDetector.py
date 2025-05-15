import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt
import math
import numpy as np
import abc  

# torch stuff
import torch

from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

from torch_nlm import nlm2d

class FeatureSqueezingDetector(metaclass=abc.ABCMeta):
    """
    Feature Squeezing Detector
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
        output = self.model(image)
        self.output = SM(output)
        for key, filter in self.prepro_dict.items():
            filtered_image = filter(image)
            self.image_dict[key] = filtered_image
            output = self.model(filtered_image)

            self.output_dict[key] = SM(output)
            self.score_dict[key] = self.calculate_score(self.output - self.output_dict[key])
        
        self.score, _ = torch.max(torch.stack(list(self.score_dict.values())), dim=0)
    
        return self.score