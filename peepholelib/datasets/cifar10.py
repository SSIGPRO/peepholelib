# Our stuff
from peepholelib.datasets.dataset_base import DatasetBase
from peepholelib.datasets.transforms import vgg16_cifar10

# torch stuff
import torch
from torch.utils.data import random_split

# CIFAR from torchvision
from torchvision import datasets

class Cifar10(DatasetBase):
    def __init__(self, **kwargs):
        '''
        Cifar10 loader (train & val & test). Validation is created from train, fixed in 0.8 for train and 0.2 for val.

        Expects:
            data_path (str): Cifar download folder. If not downloaded, downloads the dataset in this folder.
        Returns:
            - a thumbs up
        '''

        DatasetBase.__init__(self, **kwargs)

        return
    
    def __load_data__(self, **kwargs):
        '''
        Load and prepare CIFAR10 data.
        
        Args:
        - seed (int): Random seed for reproducibility.
        - transform (torchvision.transforms.Compose): Custom transform to apply to the original dataset.
        
        Returns:
        - a thumbs up
        '''
        # accepts custom transform if provided in kwargs
        transform = kwargs.get('transform',vgg16_cifar10)

        seed = kwargs.get('seed', 42)
            
        # set torch seed
        torch.manual_seed(seed)

        # Test dataset is loaded directly
        test_dataset = datasets.CIFAR10(
            root = self.data_path,
            train = False,
            transform = transform,
            download = True
        )
        
        # train data will be splitted into training and validation
        _train_data = datasets.CIFAR10( 
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
        self.__dataset__ = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
                }
        
        classes = {i: class_name for i, class_name in enumerate(train_dataset.classes)}
        self._classes = {
                'CIFAR10-train': classes,
                'CIFAR10-val': classes, 
                'CIFAR10-test': classes
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
