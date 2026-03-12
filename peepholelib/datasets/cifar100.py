# general python stuff
import pickle

# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap
<<<<<<< HEAD
=======
from peepholelib.datasets.functional.transforms import vgg16_transform
>>>>>>> 5f812a6 (banana)

# torch stuff
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import random_split, Subset

# CIFAR from torchvision
from torchvision import datasets

class CIFAR100Custom(CIFAR100):

    def __init__(self, **kwargs):
        CIFAR100.__init__(self, **kwargs)
        self.fine_to_coarse = {
            0: [4, 30, 55, 72, 95],
            1: [32, 1, 67, 73, 91],
            2: [54, 62, 70, 82, 92],
            3: [9, 10, 16, 28, 61], 
            4: [0, 51, 53, 57, 83],
            5: [22, 39, 40, 86, 87],
            6: [5, 20, 25, 84, 94],
            7: [6, 7, 14, 18, 24],
            8: [3, 42, 43, 88, 97],
            9: [12, 17, 37, 68, 76],
            10: [23, 33, 49, 60, 71], 
            11: [15, 19, 21, 31, 38],
            12: [34, 63, 64, 66, 75],
            13: [26, 45, 77, 79, 99],
            14: [2, 11, 35, 46, 98], 
            15: [27, 29, 44, 78, 93],
            16: [36, 50, 65, 74, 80],
            17: [47, 52, 56, 59, 96],
            18: [8, 13, 48, 58, 90],
            19: [41, 69, 81, 85, 89]
        }

        self.M = torch.zeros(20, 100)

        for superc, cs in self.fine_to_coarse.items():
            for c in cs:
                self.M[superc, c] = 1

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        sample = {
            "image": img,
            "label": torch.tensor(label),
            "coarse_label": self.M[:, int(label)].argmax(),
        }
        return sample

class Cifar100(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        CIFAR100 loader (train & val & test). Validation is created from the
        training split according to `train_ratio` (default: 0.8 train, 0.2 val).

        Args:
            path (str): CIFAR100 download folder. If not already available, the
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
        
<<<<<<< HEAD
        self.transform = kwargs.get('std_transform')
=======
        self.transform = kwargs.get('std_transform', vgg16_transform)
>>>>>>> 5f812a6 (banana)
        self.augmentation = kwargs.get('aug_transform', None)
        self.train_ratio = kwargs.get('train_ratio', 0.8)

        DatasetWrap.__init__(self, **kwargs)

        return
    
    def __load_data__(self):
        '''
        Load and prepare CIFAR100 data.
        
        Returns:
        - a thumbs up
        '''

        # Test dataset is loaded directly
        test_ds = CIFAR100Custom(
            root = self.path,
            train = False,
            transform = self.transform,
            download = True
        )

        base_ds = CIFAR100Custom(
                root=self.path,
                train=True,
                transform=self.transform,
                download=False
            )
        
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError(f'train_ratio must be in (0, 1), got {self.train_ratio}.')

        n_total = len(base_ds)
        n_train = int(round(n_total * self.train_ratio))
        n_val = n_total - n_train

        train_idx, val_idx = random_split(
                range(n_total),
                [n_train, n_val],
                generator=torch.Generator().manual_seed(self.seed)
            )
        
        val_ds = Subset(base_ds, val_idx)
        
        if self.augmentation is None:
                    
            train_ds = Subset(base_ds, train_idx)
        else:
            _train_aug = CIFAR100Custom(
                root=self.path,
                train=True,
                transform=self.augmentation,
                download=True
            )
            train_ds = Subset(_train_aug, train_idx)
    
        # Save datasets as objects in the class
        self.__dataset__ = {
                'CIFAR100-train': train_ds,
                'CIFAR100-val': val_ds,
                'CIFAR100-test': test_ds
                }
        
        return 

    @classmethod
    def get_classes(cls, **kwargs):
        meta_path = kwargs['meta_path']
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        labels = {i: name for i, name in enumerate(meta['fine_label_names'])}
        return labels 
    
    def get_superclasses(cls):

        return {0: 'aquatic mammals',
                1: 'fish',
                2: 'flowers',
                3: 'food containers',
                4: 'fruit and vegetables',
                5: 'household electrical devices',
                6: 'household furniture',
                7: 'insects',
                8: 'large carnivores',
                9: 'large man-made outdoor things',
                10: 'large natural outdoor scenes',
                11: 'large omnivores and herbivores',
                12: 'medium-sized mammals',
                13: 'non-insect invertebrates',
                14: 'people',
                15: 'reptiles',
                16: 'small mammals',
                17: 'trees',
                18: 'vehicles 1',
                19: 'vehicles 2'}
