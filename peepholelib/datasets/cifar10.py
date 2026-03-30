# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap

# torch stuff
import torch
from torch.utils.data import random_split, Subset

# CIFAR from torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor 

class CIFAR10Custom(CIFAR10):
    def __init__(self, **kwargs):
        CIFAR10.__init__(self, **kwargs)
        self._to_tensor = ToTensor()

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        sample = {
                "image": self._to_tensor(img),
                "label": torch.tensor(label),
            }
        return sample

class Cifar10(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        CIFAR10 loader (train & val & test). Validation is created from the training split according to `train_ratio` (default: 0.8 train, 0.2 val).

        Args:
            path (str): CIFAR10 download folder. If not already available, the dataset is downloaded in this folder. images.
            augmentation (callable, optional): If provided, applied only to the training split.
            train_ratio (float, optional): Fraction of training samples used for train (remainder goes to val).
            seed (int, optional): Random seed used for deterministic train/val splitting.
        Returns:
            - a thumbs up
        '''
        DatasetWrap.__init__(self, **kwargs)

        self.transform = kwargs.get('std_transform')
        self.augmentation = kwargs.get('aug_transform', None)
        self.train_ratio = kwargs.get('train_ratio', 0.8)

        # append ToTensor to the transform
        if self.transform != None:
            self.transform.transforms.append(ToTensor())
        else:
            self.transform = ToTensor()
                                                                          
        # if augmentation == None, transform will be used for all loaders
        if self.augmentation != None:
            self.augmentation.transforms.append(ToTensor())

        return
    
    def __load_data__(self, **kwargs):
        '''
        Load and prepare CIFAR10 data.
        
        Args:
        - seed (int): Random seed for reproducibility.
        
        Returns:
        - a thumbs up
        '''
        # Test dataset is loaded directly
        test_ds = CIFAR10Custom(
            root = self.path,
            train = False,
            download = True
        )

        # train is splitted into train and val
        base_ds = CIFAR10Custom(
                root=self.path,
                train=True,
                download=False
            )
        
        train_ds, val_ds = random_split(
                base_ds,
                [self.train_ratio, 1 - self.train_ratio],
                generator=torch.Generator().manual_seed(self.seed)
            )
        
        # Save datasets as objects in the class
        self.__dataset__ = {
                'train': train_ds,
                'val': val_ds,
                'test': test_ds
                }
        
        classes = {i: class_name for i, class_name in enumerate(train_ds.classes)}
        self._classes = {
                'CIFAR10-train': classes,
                'CIFAR10-val': classes, 
                'CIFAR10-test': classes
                }
        
        return 
