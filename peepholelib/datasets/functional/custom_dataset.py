# Our stuff
from peepholelib.datasets.dataset_base import DatasetBase
# General python stuff
from pathlib import Path as Path

# torch stuff
import torch
from torch.utils.data import random_split

# CIFAR from torchvision
from torchvision import datasets

class CustomDataset(DatasetBase):
    def __init__(self, **kwargs):
        '''
        Custom dataset loader. Validation is created from train, fixed in 0.8 for train and 0.2 for val.

        Expects:
            data_path (str): Custom dataset folder.
        Returns:
            - a thumbs up
        '''

        DatasetBase.__init__(self, **kwargs)
        
        return
    
    def load_data(self, **kwargs):
        '''
        Load a custom dataset for activations, corevectors, peepholes, conceptograms.
        
        Returns:
        - a thumbs up
        '''

        # Load the dataset from the specified path
        self._dss = kwargs['dataset'] if 'dataset' in kwargs else None
        
        # Set classes
        self._classes = kwargs['classes'] if 'classes' in kwargs else None
        
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