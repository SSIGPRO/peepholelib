# general python stuff
import pickle

# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.functional.transforms import vgg16_cifar100

# torch stuff
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import random_split

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
        img, target = super().__getitem__(index)

        return torch.tensor(img), torch.tensor(target), torch.tensor(self.M[:, target].argmax())

class Cifar100(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        Cifar100 loader (train & val & test). Validation is created from train, fixed in 0.8 for train and 0.2 for val.

        Args:
            path (str): Cifar download folder. If not downloaded, downloads the dataset in this folder.
        Returns:
            - a thumbs up
        '''
        
        # add a default transform for specific DS
        if 'transform' not in kwargs:
            kwargs['transform'] = vgg16_cifar100

        DatasetWrap.__init__(self, **kwargs)

        return
    
    def __load_data__(self):
        '''
        Load and prepare CIFAR100 data.
        
        Returns:
        - a thumbs up
        '''
        
        transform = self.transform
        seed = self.seed
            
        # set torch seed
        torch.manual_seed(seed)

        # Test dataset is loaded directly
        test_dataset = CIFAR100Custom(
            root = self.path,
            train = False,
            transform = transform,
            download = True
        )
        
        # train data will be splitted into training and validation
        _train_data = CIFAR100Custom( 
            root = self.path,
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
                'CIFAR100-train': train_dataset,
                'CIFAR100-val': val_dataset,
                'CIFAR100-test': test_dataset
                }
        
        return 

    @classmethod
    def get_classes(cls, **kwargs):
        meta_path = kwargs['meta_path']
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        labels = {i: name for i, name in enumerate(meta['fine_label_names'])}
        return labels 
    
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
