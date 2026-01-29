# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.functional.transforms import vgg16
from torchvision.datasets import Places365 as torchPlaces

# torch stuff
import torch
from torch.utils.data import random_split

# Places365 from torchvision
from torchvision import datasets

class PlacesCustom(torchPlaces):

    def __init__(self, **kwargs):
        torchPlaces.__init__(self, **kwargs)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        return {'image': img,
                'label': torch.tensor(label),
                }  

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
            kwargs['transform'] = vgg16

        self.splitting_ratio = kwargs.get('splitting_ratio', [0.45205478, 0.27397261, 0.27397261])

        DatasetWrap.__init__(self, **kwargs)

        return
    
    def __load_data__(self):
        '''
        Load and prepare Places data.
        
        Returns:
        - a thumbs up
        '''

        transform = self.transform
        target_transform = self.target_transform
        seed = self.seed              

        # set torch seed
        torch.manual_seed(seed)

        _data = PlacesCustom(
                root = self.path,
                split = 'val',
                transform = transform,
                small = True,
                download = True
                )

        # split to get 10000 samples for val and test
        _, val_dataset, test_dataset = random_split(
                _data,
                self.splitting_ratio, # to get exactly 10000 samples
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
