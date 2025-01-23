# Our stuff
from .dataset_base import DatasetBase

# General python stuff
from pathlib import Path as Path

# torch stuff
import torch
from torch.utils.data import random_split, DataLoader

# CIFAR from torchvision
from torchvision import transforms, datasets
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

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
        Load and prepare data for a specified portion of a dataset.
        
        Args:
        - dataset (str): The name of the dataset ('CIFAR10', 'CIFAR100').
        - batch_size (int): The batch size for DataLoader.
        - data_kwargs (dict): Additional keyword arguments for DataLoader.
        - seed (int): Random seed for reproducibility.
        - data_augmentation (bool): Flag indicating whether to apply data 
        augmentation (default: False).
        - original_transform (torchvision.transforms.Compose): Custom transform to apply to the original dataset. (default: CIFAR10/CIFAR100 transform)
        - augmentation_transform (torchvision.transforms.Compose): Custom transform to apply to the augmented dataset. (default: CIFAR10/CIFAR100 transform)
        
        Returns:
        - dict: containing a DataLoader for 'train', 'val', 'test', and a dictionary mapping class indices to class names for 'classes'.
        
        Example:
        - To load the training data of CIFAR10 with a batch size of 32:
        >>> c = Cifar(dataset = 'CIFAR10')
        >>> loaders = c.load_data(batch_size=32, data_kwargs={}, seed=42)

        To get a dictionary mapping class indices to names:
        >>> class_dict = loaders['classes']
        
        To get the train, validation, and data:
        >>> train_data = loaders['train']
        >>> train_data = loaders['val']
        >>> train_data = loaders['test']
        '''

        # parse parameteres
        batch_size = kwargs['batch_size']
        data_kwargs = kwargs['data_kwargs']
        seed = kwargs['seed']

        data_augmentation = kwargs['data_augmentation'] if 'data_augmentation' in kwargs else False

        # set torch seed
        torch.manual_seed(seed)

        # original dataset without augmentation
        # accepts custom transform if provided in kwargs
        if 'original_transform' in kwargs: original_transform = kwargs['original_transform']
            
        # default transform (CIFAR10 / CIFAR100)    
        else: 
            if self.dataset == 'CIFAR10': normalize_transform = transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
            else: normalize_transform = transforms.Normalize((0.438, 0.418, 0.377), (0.300, 0.287, 0.294)) # CIFAR100
            original_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize_transform
            ])

        # Test dataset is loaded directly
        test_dataset = datasets.__dict__[self.dataset](
            root=self.data_path,
            train=False,
            transform=original_transform,
            download=True
        )
        
        # train data will be splitted for training and validation
        _train_data = datasets.__dict__[self.dataset]( 
            root=self.data_path,
            train=True,
            transform=None, #original_transform,
            download=True
        )
        
        train_dataset, val_dataset = random_split(
            _train_data,
            [0.8, 0.2],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # set validation dataset transform
        val_dataset.dataset.transform = original_transform
        
        # Apply the transformation according to data augmentation 
        if data_augmentation:
            # accepts custom augmentation transform if provided in kwargs
            # default transform is CIFAR10 / CIFAR100 AutoAugment
            if self.dataset == 'CIFAR10': normalize_transform = transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
            else: normalize_transform = transforms.Normalize((0.438, 0.418, 0.377), (0.300, 0.287, 0.294)) # CIFAR100
            augmentation_transform = kwargs.get('augmentation_transform', transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10), 
                transforms.ToTensor(),
                normalize_transform
            ]))
            train_dataset.dataset.transform = augmentation_transform 
        else:
            train_dataset.dataset.transform = original_transform
     
        # Save datasets as objects in the class
        self._train_ds = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, **data_kwargs)
        self._val_ds = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **data_kwargs)
        self._test_ds = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **data_kwargs)
        self._classes = {i: class_name for i, class_name in enumerate(test_dataset.classes)}  
        
        self._loaders = {
            'train': self._train_ds,
            'val': self._val_ds,
            'test': self._test_ds
            }

        return self._loaders
