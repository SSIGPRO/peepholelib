# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap
from torchvision.datasets import SVHN as torchSVHN

# torch stuff
import torch
from torch.utils.data import random_split

# SVHN from torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor 

class SVHNCustom(torchSVHN):

    def __init__(self, **kwargs):
        torchSVHN.__init__(self, **kwargs)
        self._to_tensor = ToTensor()

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        return {
                'image': self._to_tensor(img),
                'label': torch.tensor(label),
                }  

class SVHN(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        SVHN loader (train & val & test). Validation is created from train, fixed in 0.8 for train and 0.2 for val.

        Expects:
            path (str): SVHN download folder. If not downloaded, downloads the dataset in this folder.
        Returns:
            - a thumbs up
        '''
        self.train_ratio = kwargs.get('train_ratio', 0.86349)
        self.test_ratio = kwargs.get('test_ratio', 0.38415)

        DatasetWrap.__init__(self, **kwargs)

        return
    
    def __load_data__(self):
        '''
        Load and prepare SVHN data.
        
        Args:
        - seed (int): Random seed for reproducibility.
        
        Returns:
        - a thumbs up
        '''

        # split to get 10000 samples for test
        _test_data = SVHNCustom(
            root = self.path,
            split = 'test',
            download = True
        )

        _, test_dataset = random_split(
                _test_data,
                [1 - self.test_ratio, self.test_ratio],
                generator=torch.Generator().manual_seed(self.seed)
                )
        
        # split to get 10000 samples for val
        _train_data = SVHNCustom( 
            root = self.path,
            split = 'train',
            download = True
        )
        
        train_dataset, val_dataset = random_split(
                _train_data,
                [self.train_ratio, 1 - self.train_ratio],
                generator=torch.Generator().manual_seed(seed)
                )

        self.__dataset__ = {
                'SVHN-train': train_dataset,
                'SVHN-val': val_dataset,
                'SVHN-test': test_dataset
                }
        
        # TODO: implement get_classes()
        #self._classes = {
        #        'SVHN-train': {i: class_name for i, class_name in enumerate(train_dataset.classes)},
        #        'SVHN-val': {i: class_name for i, class_name in enumerate(val_dataset.classes)},
        #        'SVHN-test': {i: class_name for i, class_name in enumerate(test_dataset.classes)}
        #        }

        return 
