# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap

# torch stuff
import torch
from torch.utils.data import random_split

# Places365 from torchvision
from torchvision.datasets import Places365 as torchPlaces
from torchvision.transforms import ToTensor

class PlacesCustom(torchPlaces):

    def __init__(self, **kwargs):
        torchPlaces.__init__(self, **kwargs)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        return {
                'image': img,
                'label': torch.tensor(label),
                }  

class Places(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        Places loader (val & test). Validation is created from train.

        Expects:
            path (str): Places download folder. If not downloaded, downloads the dataset in this folder.
        Returns:
            - a thumbs up
        '''
        DatasetWrap.__init__(self, **kwargs)

        # add a default transform for specific DS
        self.transform = kwargs.get('std_transform', None)

<<<<<<< HEAD
        self.splitting_ratio = kwargs.get('splitting_ratio', [0.45205478, 0.27397261, 0.27397261])

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
        Load and prepare Places data.
        
        Returns:
        - a thumbs up
        '''
        try:
            _data = PlacesCustom(
                    root = self.path,
                    split = 'val',
                    transform = self.transform,
                    small = True,
                    download = False 
                    )
        except RuntimeError:
            print("Seems like Places is not downloaded. Will download it.") 
            _data = PlacesCustom(
                    root = self.path,
                    split = 'val',
                    transform = self.transform,
                    small = True,
                    download = True 
                    )


        # split to get 10000 samples for val and test
        _, val_dataset, test_dataset = random_split(
                _data,
                self.splitting_ratio, 
                generator=torch.Generator().manual_seed(self.seed)
                )
                    
        self.__dataset__ = {
                'Places365-val': val_dataset,
                'Places365-test': test_dataset
                }
        
        return
