# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap

# torch stuff
import torch
from torch.utils.data import random_split

# MNIST from torchvision
from torchvision.datasets import MNIST as torchMNIST
from torchvision.transforms import ToTensor

class MNISTCustom(torchMNIST):
    def __init__(self, **kwargs):
        torchMNIST.__init__(self, **kwargs)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        return {
                'image': img,
                'label': torch.tensor(label),
                }  

class MNIST(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        MNist loader (val & test). Validation is created from train.

        Expects:
            path (str): MNIST download folder. If not downloaded, downloads the dataset in this folder.
        Returns:
            - a thumbs up
        '''
        DatasetWrap.__init__(self, **kwargs)

        # add a default transform for specific DS
        self.transform = kwargs.get('std_transform', None)

        self.splitting_ratio = kwargs.get('splitting_ratio', [5/6, 1/6])

        # append ToTensor to the transform
        if self.transform != None:
            self.transform.transforms.append(ToTensor())
        else:
            self.transform = ToTensor()

        return
    
    def __load_data__(self):
        '''
        Load and prepare MNIST data.
        
        Returns:
        - a thumbs up
        '''

        _train_data = MNISTCustom(
                root = self.path,
                train = True,
                transform = self.transform,
                download = True
                )

        # split to get 10000 samples for val and test
        _, val_dataset = random_split(
                _train_data,
                self.splitting_ratio, 
                generator=torch.Generator().manual_seed(self.seed)
                )

        # this one has 10000 samples for test
        test_dataset = MNISTCustom(
                root = self.path,
                train = False,
                transform = self.transform,
                download = True
                )
                    
        self.__dataset__ = {
                'MNIST-val': val_dataset,
                'MNIST-test': test_dataset
                }

        return
