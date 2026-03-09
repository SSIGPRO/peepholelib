# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap

# torch stuff
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, Subset

# CIFAR from torchvision
from torchvision import datasets

class CIFAR10Custom(CIFAR10):
    def __init__(self, **kwargs):
        CIFAR10.__init__(self, **kwargs)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        sample = {
                "image": img,
                "label": torch.tensor(label),
            }
        return sample

class Cifar10(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        CIFAR10 loader (train & val & test). Validation is created from the
        training split according to `train_ratio` (default: 0.8 train, 0.2 val).

        Args:
            path (str): CIFAR10 download folder. If not already available, the
                dataset is downloaded in this folder.
            transform (callable, optional): Transform applied to validation/test
                images. Defaults to `vgg16`.
            augmentation (callable, optional): If provided, applied only to the
                training split.
            train_ratio (float, optional): Fraction of training samples used for
                train (remainder goes to val).
            seed (int, optional): Random seed used for deterministic train/val
                splitting.
        Returns:
            - a thumbs up
        '''

        self.transform = kwargs.get('std_transform')
        self.augmentation = kwargs.get('aug_transform', None)
        self.train_ratio = kwargs.get('train_ratio', 0.8)

        DatasetWrap.__init__(self, **kwargs)

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

        # Test dataset is loaded directly
        test_ds = CIFAR10Custom(
            root = self.path,
            train = False,
            transform = self.transform,
            download = True
        )

        base_ds = CIFAR10Custom(
                root=self.path,
                train=True,
                transform=self.transform,
                download=False
            )
        
        train_idx, val_idx = random_split(
                range(len(base_ds)),
                [self.train_ratio, 1 - self.train_ratio],
                generator=torch.Generator().manual_seed(self.seed)
            )
        
        val_ds = Subset(base_ds, val_idx)
        
        if self.augmentation is None:
                    
            train_ds = Subset(base_ds, train_idx)

        else:
            _train_aug = CIFAR10Custom(
                root=self.path,
                train=True,
                transform=self.augmentation,
                download=True
            )
            train_ds = Subset(_train_aug, train_idx)
    
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
