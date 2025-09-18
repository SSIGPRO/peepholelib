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

# CIFAR from torchvision
from torchvision import datasets
from PIL import Image

class CustomDS(Dataset):
    def __init__(self, data, labels, transform):
        Dataset.__init__(self) 
        self.data = []
        for d in tqdm(data, disable=True):
            self.data.append(Image.fromarray(d))
        self.labels = labels
        self.transform = transform
        self.len = labels.shape[0]
        return

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        d = self.transform(self.data[idx])
        l = self.labels[idx]
        return d, l

    def __getitems__(self, idxs):
        return [(self.transform(self.data[i]), self.labels[i]) for i in idxs]

class Places(DatasetBase):
    def __init__(self, **kwargs):
        '''
        Places loader (train & val & test). Validation is created from train, fixed in 0.8 for train and 0.2 for val.

        Expects:
            data_path (str): Places download folder. If not downloaded, downloads the dataset in this folder.
        Returns:
            - a thumbs up
        '''

        DatasetBase.__init__(self, **kwargs)
        
        # use Places by default
        self.dataset = kwargs.get('dataset', 'Places')

        # raise error if the dataset is not Places
        if "places" not in self.dataset.lower():
            raise ValueError("Dataset must be Places365")

        return
    
    def __load_data__(self, **kwargs):
        '''
        Load and prepare Places data.
        
        Args:
        - seed (int): Random seed for reproducibility.
        - transform (torchvision.transforms.Compose): Custom transform to apply to the original dataset. (default: Places for vgg16 transform)
        
        Returns:
        - a thumbs up
        '''
        # accepts custom transform if provided in kwargs
        transform = kwargs.get('transform', eval('vgg16_'+self.dataset.lower()))

        seed = kwargs.get('seed', 42)
            
        # set torch seed
        torch.manual_seed(seed)

        train_dataset = datasets.__dict__[self.dataset]( 
                        root = self.data_path,
                        split = 'train',
                        transform = transform,
                        small = True,
                        download = True
                    )

        val_set = datasets.__dict__[self.dataset]( 
                        root = self.data_path,
                        split = 'val',
                        transform = None,
                        small = True,
                        download = True
                    )
                    
        val_dataset, test_dataset = random_split(
            val_set,
            [0.5, 0.5], 
            generator=torch.Generator().manual_seed(seed)
        )

        # Apply the transform 
        if transform != None:
            val_dataset.dataset.transform = transform
            test_dataset.dataset.transform = transform

        self.__dataset__ = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
                }
        
        self._classes = {
                'Places-train': {i: class_name for i, class_name in enumerate(train_dataset.classes)},
                'Places-val': {i: class_name for i, class_name in enumerate(val_dataset.classes)},
                'Places-test': {i: class_name for i, class_name in enumerate(test_dataset.classes)}
                }

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
        if not self.__dataset__:
            raise RuntimeError('Data not loaded. Please run load_data() first.')
        
        return [self.__dataset__[ds_key][idx]]
