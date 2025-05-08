# Our stuff
from .dataset_base import DatasetBase
from .transforms import vgg16_cifar100

# General python stuff
from pathlib import Path as Path

# torch stuff
import torch
from torch.utils.data import random_split, DataLoader

# CIFAR from torchvision
from torchvision import transforms, datasets


class Cifar(DatasetBase):
    def __init__(self, **kwargs):
        DatasetBase.__init__(self, **kwargs)
        
        # use CIFAR10 by default
        if 'dataset' in kwargs:
            self.dataset = kwargs['dataset']
        else:
            self.dataset = 'CIFAR10'
        print('dataset: %s' % self.dataset)

        # raise error if the dataset is not CIFAR
        if "cifar" not in self.dataset.lower():
            raise ValueError("Dataset must be CIFAR")

        '''
        CIFAR10 num_classes: 10
        CIFAR100 num_classes: 100
        '''
        return
    
    def load_data(self, **kwargs):
        '''
        Load and prepare CIFAR10 or CIFAR100 data.
        
        Args:
        - seed (int): Random seed for reproducibility.
        - transform (torchvision.transforms.Compose): Custom transform to apply to the original dataset. (default: CIFAR10/CIFAR100 transform)
        
        Returns:
        - a thumbs up
        '''

        # parse parameteres
        seed = kwargs['seed']

        # original dataset without augmentation
        # accepts custom transform if provided in kwargs
        transform = kwargs['transform'] if 'transform' in kwargs else vgg16_cifar100
            
        # set torch seed
        torch.manual_seed(seed)

        # Test dataset is loaded directly
        test_dataset = datasets.__dict__[self.dataset](
            root = self.data_path,
            train = False,
            transform = transform,
            download = True
        )
        
        # train data will be splitted for training and validation
        _train_data = datasets.__dict__[self.dataset]( 
            root = self.data_path,
            train = True,
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
     
        # Save datasets as objects in the class
        self._dss = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
                }
        self._classes = {i: class_name for i, class_name in enumerate(test_dataset.classes)}  
        
        return 
