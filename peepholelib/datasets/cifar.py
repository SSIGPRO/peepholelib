# Our stuff
from peepholelib.datasets.dataset_base import DatasetBase
from peepholelib.datasets.transforms import vgg16_cifar10, vgg16_cifar100

# General python stuff
from pathlib import Path as Path

# torch stuff
import torch
from torch.utils.data import random_split

# CIFAR from torchvision
from torchvision import datasets


class Cifar(DatasetBase):
    def __init__(self, **kwargs):
        '''
        Cifar loader (train & val & test). Validation is created from train, fixed in 0.8 for train and 0.2 for val.

        Expects:
            data_path (str): Cifar download folder. If not downloaded, downloads the dataset in this folder.
        Returns:
            - a thumbs up
        '''

        DatasetBase.__init__(self, **kwargs)
        
        # use CIFAR10 by default
        self.dataset = kwargs['dataset'] if 'dataset' in kwargs else 'CIFAR10'
        print('dataset: %s' % self.dataset)

        # raise error if the dataset is not CIFAR
        if "cifar" not in self.dataset.lower():
            raise ValueError("Dataset must be CIFAR")

        return
    
    def load_data(self, **kwargs):
        '''
        Load and prepare CIFAR10 or CIFAR100 data.
        
        Args:
        - seed (int): Random seed for reproducibility.
        - transform (torchvision.transforms.Compose): Custom transform to apply to the original dataset. (default: CIFAR10/CIFAR100 for vgg16 transform)
        
        Returns:
        - a thumbs up
        '''
        # accepts custom transform if provided in kwargs
        transform = kwargs['transform'] if 'transform' in kwargs else eval('vgg16_'+self.dataset.lower())
            
        seed = kwargs['seed']
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