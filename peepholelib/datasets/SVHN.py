# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.functional.transforms import vgg16_svhn

# torch stuff
import torch
from torch.utils.data import random_split

# SVHN from torchvision
from torchvision import datasets

class SVHN(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        SVHN loader (train & val & test). Validation is created from train, fixed in 0.8 for train and 0.2 for val.

        Expects:
            path (str): SVHN download folder. If not downloaded, downloads the dataset in this folder.
        Returns:
            - a thumbs up
        '''
        
        # add a default transform for specific DS
        if 'transform' not in kwargs:
            kwargs['transform'] = vgg16_svhn

        DatasetWrap.__init__(self, **kwargs)

        return
    
    def __load_data__(self):
        '''
        Load and prepare SVHN data.
        
        Args:
        - seed (int): Random seed for reproducibility.
        - transform (torchvision.transforms.Compose): Custom transform to apply to the original dataset.
        
        Returns:
        - a thumbs up
        '''
        transform = self.transform
        seed = self.seed 

        # set torch seed
        torch.manual_seed(seed)

        # split to get 10000 samples for test
        _test_data = datasets.__dict__['SVHN'](
            root = self.path,
            split = 'test',
            transform = transform,
            download = True
        )

        _, test_dataset = random_split(
                _test_data,
                [0.61585, 0.38415],
                generator=torch.Generator().manual_seed(seed)
                )
        
        # split to get 10000 samples for val
        _train_data = datasets.__dict__['SVHN']( 
            root = self.path,
            split = 'train',
            transform = transform,
            download = True
        )
        
        _, val_dataset = random_split(
                _train_data,
                [0.86349, 0.13651],
                generator=torch.Generator().manual_seed(seed)
                )

        self.__dataset__ = {
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
