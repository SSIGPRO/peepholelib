# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap

# torch stuff
import torch
from torchvision.datasets import DTD
from torchvision.transforms import ToTensor, Compose, Resize


class TexturesCustom(DTD):

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return {
            'image': img,
            'label': torch.tensor(label),
        }


class Textures(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        Textures (DTD) Far-OOD loader (val & test). Uses the official DTD val and test partitions.
        Used as a Far-OOD dataset in the OpenOOD benchmark.

        Args:
            path (str): DTD download folder. If not already available, the dataset is downloaded here.
            transform (callable, optional): Transform applied to images. Defaults to Resize(224) + ToTensor.
            seed (int, optional): Random seed (not used for splitting; DTD has official splits).
        Returns:
            - a thumbs up
        '''
        DatasetWrap.__init__(self, **kwargs)

        self.transform = kwargs.get('std_transform', None)

        if self.transform is not None:
            self.transform.transforms.append(ToTensor())
        else:
            self.transform = Compose([Resize((224, 224)), ToTensor()])

        return

    def __load_data__(self):
        val_dataset = TexturesCustom(
            root=self.path,
            split='val',
            transform=self.transform,
            download=True,
        )

        test_dataset = TexturesCustom(
            root=self.path,
            split='test',
            transform=self.transform,
            download=True,
        )

        self.__dataset__ = {
            'Textures-val': val_dataset,
            'Textures-test': test_dataset,
        }

        return
