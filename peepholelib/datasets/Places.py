# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.functional.transforms import vgg16_imagenet

# torch stuff
import torch
from torch.utils.data import random_split

# CIFAR from torchvision
from torchvision import datasets

class Places(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        Places loader (train & val & test). Validation is created from train, fixed in 0.8 for train and 0.2 for val.

        Expects:
            path (str): Places download folder. If not downloaded, downloads the dataset in this folder.
        Returns:
            - a thumbs up
        '''

        # add a default transform for specific DS
        if 'transform' not in kwargs:
            kwargs['transform'] = vgg16_imagenet

        DatasetWrap.__init__(self, **kwargs)

        return
    
    def __load_data__(self):
        '''
        Load and prepare Places data.
        
        Returns:
        - a thumbs up
        '''

        transform = self.transform
        seed = self.seed              

        # set torch seed
        torch.manual_seed(seed)

        train_dataset = datasets.__dict__[self.dataset]( 
                        root = self.path,
                        split = 'train',
                        transform = transform,
                        small = True,
                        download = True
                    )

        val_set = datasets.__dict__[self.dataset]( 
                        root = self.path,
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
