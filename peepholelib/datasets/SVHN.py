# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.functional.transforms import vgg16_svhn

# torch stuff
import torch

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

        # Test dataset is loaded directly
        test_dataset = datasets.__dict__[self.dataset](
            root = self.path,
            split = 'test',
            transform = transform,
            download = True
        )
        
        # train data will be splitted into training and validation
        _train_data = datasets.__dict__[self.dataset]( 
            root = self.path,
            split = 'train',
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

        self.__dataset__ = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
                }
        
        self._classes = {
                'SVHN-train': {i: class_name for i, class_name in enumerate(train_dataset.classes)},
                'SVHN-val': {i: class_name for i, class_name in enumerate(val_dataset.classes)},
                'SVHN-test': {i: class_name for i, class_name in enumerate(test_dataset.classes)}
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
