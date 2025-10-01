# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.functional.transforms import vgg16_imagenet

# torch stuff
import torch
from torch.utils.data import random_split

# Places365 from torchvision
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

        _data = datasets.__dict__['Places365'](
                root = self.path,
                split = 'val',
                transform = transform,
                small = True,
                download = True
                )

        # split to get 10000 samples for val and test
        _, val_dataset, test_dataset = random_split(
                _data,
                [0.45205478, 0.27397261, 0.27397261], # to get exactly 10000 samples
                generator=torch.Generator().manual_seed(seed)
                )
                    
        self.__dataset__ = {
                'Places365-val': val_dataset,
                'Places365-test': test_dataset
                }
        
        # TODO: implement get_classes()
        #self._classes = {
        #        'Places-train': {i: class_name for i, class_name in enumerate(train_dataset.classes)},
        #        'Places-val': {i: class_name for i, class_name in enumerate(val_dataset.classes)},
        #        'Places-test': {i: class_name for i, class_name in enumerate(test_dataset.classes)}
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
