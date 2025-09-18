# Our stuff
from peepholelib.datasets.dataset_base import DatasetBase
from peepholelib.datasets.transforms import vgg16_cifar10, vgg16_cifar100

# General python stuff
from pathlib import Path as Path
import numpy as np
from math import floor
from tqdm import tqdm

# torch stuff
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset

# SVHN from torchvision
from torchvision import datasets
from PIL import Image

class SVHN(DatasetBase):
    def __init__(self, **kwargs):
        '''
        SVHN loader (train & val & test). Validation is created from train, fixed in 0.8 for train and 0.2 for val.

        Expects:
            data_path (str): SVHN download folder. If not downloaded, downloads the dataset in this folder.
        Returns:
            - a thumbs up
        '''

        DatasetBase.__init__(self, **kwargs)
        
        # use SVHN by default
        self.dataset = kwargs.get('dataset', 'CIFAR10')

        # raise error if the dataset is not SVHN
        if "cifar" not in self.dataset.lower():
            raise ValueError("Dataset must be SVHN<10|100>")

        return
    
    def __load_data__(self, **kwargs):
        '''
        Load and prepare SVHN or SVHN data.
        
        Args:
        - seed (int): Random seed for reproducibility.
        - transform (torchvision.transforms.Compose): Custom transform to apply to the original dataset. (default: SVHN for vgg16 transform)
        
        Returns:
        - a thumbs up
        '''
        # accepts custom transform if provided in kwargs
        transform = kwargs.get('transform', eval('vgg16_'+self.dataset.lower()))

        seed = kwargs.get('seed', 42)
            
        # set torch seed
        torch.manual_seed(seed)

        # Test dataset is loaded directly
        test_dataset = datasets.__dict__[self.dataset](
            root = self.data_path,
            split = 'test',
            transform = transform,
            download = True
        )
        
        # train data will be splitted into training and validation
        train_dataset = datasets.__dict__[self.dataset]( 
            root = self.data_path,
            split = 'train',
            transform = None, #transform,
            download = True
        )
        
        train_dataset, val_dataset = random_split(
            _train_data,
            [0.8, 0.2],
            generator=torch.Generator().manual_seed(seed)
        )

        # Apply the transform 
        if transform != None:
            val_dataset.dataset.transform = transform
            train_dataset.dataset.transform = transform 
        
        return 
    
    def get(self, ds_key, idx):
        '''
        Get item from the dataset.
        
        Args:
        - idx (int): Index of the item to get.
        - ds_key (str): Key of the dataset to get the item from ('train', 'val', 'test').
        
        Returns:
        - a tuple of (image, label)
        '''
        if not self._dss:
            raise RuntimeError('Data not loaded. Please run load_data() first.')
        
        return [self._dss[ds_key][idx]]
