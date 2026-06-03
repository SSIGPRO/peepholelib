# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap

# torch stuff
import torch

# Textures from torchvision
from torchvision.datasets import DTD as torchTextures 
from torchvision.transforms import ToTensor, Resize, Compose

class TexturesCustom(torchTextures):
    def __init__(self, **kwargs):
        torchTextures.__init__(self, **kwargs)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        return {
                'image': img,
                'label': torch.tensor(label),
                }  

class Textures(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        Wrap for Describable Textures Dataset
        https://www.robots.ox.ac.uk/~vgg/data/dtd/ 

        Textures loader (val & test).

        Expects:
            path (str): Textures download folder. If not downloaded, downloads the dataset in this folder.
        Returns:
            - a thumbs up
        '''
        DatasetWrap.__init__(self, **kwargs)

        # add a default transform for specific DS
        self.transform = kwargs.get('std_transform', None)

        # append ToTensor to the transform
        if self.transform != None:
            self.transform.transforms.append(ToTensor())
        else:
            self.transform = Compose([ToTensor(), Resize((224, 224))])

        return
    
    def __load_data__(self):
        '''
        Load and prepare Textures data.
        
        Returns:
        - a thumbs up
        '''

        val_dataset = TexturesCustom(
                root = self.path,
                split = 'val',
                transform = self.transform,
                download = True
                )

        test_dataset = TexturesCustom(
                root = self.path,
                split = 'test',
                transform = self.transform,
                download = True
                )

        self.__dataset__ = {
                'Textures-val': val_dataset,
                'Textures-test': test_dataset
                }

        return
