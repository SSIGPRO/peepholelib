# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap

# torch stuff
import torch
from torch.utils.data import random_split

# SVHN from torchvision
from torchvision.datasets import SVHN as torchSVHN
from torchvision.transforms import ToTensor

class SVHNCustom(torchSVHN):

    def __init__(self, **kwargs):
        torchSVHN.__init__(self, **kwargs)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        return {'image': img,
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
        DatasetWrap.__init__(self, **kwargs)
        
        # add a default transform for specific DS
        self.transform = kwargs.get('std_transform', None)

<<<<<<< HEAD
        self.train_ratio = kwargs.get('train_ratio', 0.86349)
        self.test_ratio = kwargs.get('test_ratio', 0.38415)

        # append ToTensor to the transform
        if self.transform != None:
            self.transform.transforms.append(ToTensor())
        else:
            self.transform = ToTensor()
=======
        # make labels tensors unless caller explicitly overrides
        if 'target_transform' not in kwargs:
            kwargs['target_transform'] = lambda y: torch.as_tensor(y, dtype=torch.long)

        DatasetWrap.__init__(self, **kwargs)
>>>>>>> ec6a98a (starting back on XAI (#122))

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
<<<<<<< HEAD
=======
        transform = self.transform
        target_transform = self.target_transform
        seed = self.seed 

        # set torch seed
        torch.manual_seed(seed)
>>>>>>> ec6a98a (starting back on XAI (#122))

        # split to get 10000 samples for test
        _test_data = SVHNCustom(
            root = self.path,
            split = 'test',
<<<<<<< HEAD
            transform = self.transform,
=======
            transform = transform,
            target_transform = target_transform,
>>>>>>> ec6a98a (starting back on XAI (#122))
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
<<<<<<< HEAD
            transform = self.transform,
=======
            transform = transform,
            target_transform = target_transform,
>>>>>>> ec6a98a (starting back on XAI (#122))
            download = True
        )
        
        _, val_dataset = random_split(
                _train_data,
                [self.train_ratio, 1 - self.train_ratio],
                generator=torch.Generator().manual_seed(self.seed)
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
