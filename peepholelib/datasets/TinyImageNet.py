# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap

# torch stuff
import torch
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize


class TinyImageNetCustom(ImageFolder):

    def __init__(self, **kwargs):
        ImageFolder.__init__(self, **kwargs)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return {
            'image': img,
            'label': torch.tensor(label),
        }


class TinyImageNet(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        TinyImageNet Near-OOD loader (val & test). Used as a Near-OOD dataset in the OpenOOD benchmark for CIFAR100.
        Data is expected in ImageFolder format (one subdirectory per class).
        TinyImageNet is not available via torchvision and must be downloaded manually, e.g. from
        http://cs231n.stanford.edu/tiny-imagenet-200.zip

        Args:
            path (str): Path to the TinyImageNet folder (ImageFolder-compatible layout).
            transform (callable, optional): Transform applied to images. Defaults to Resize(32) + ToTensor.
            splitting_ratio (list[float], optional): [val, test] fractions. Must sum to 1.0. Defaults to [0.5, 0.5].
            seed (int, optional): Random seed for deterministic splits.
        Returns:
            - a thumbs up
        '''
        DatasetWrap.__init__(self, **kwargs)

        self.transform = kwargs.get('std_transform', None)
        self.splitting_ratio = kwargs.get('splitting_ratio', [0.5, 0.5])

        if self.transform is not None:
            self.transform.transforms.append(ToTensor())
        else:
            self.transform = Compose([Resize((32, 32)), ToTensor()])

        return

    def __load_data__(self):
        _data = TinyImageNetCustom(
            root=self.path,
            transform=self.transform,
        )

        val_dataset, test_dataset = random_split(
            _data,
            self.splitting_ratio,
            generator=torch.Generator().manual_seed(self.seed)
        )

        self.__dataset__ = {
            'TinyImageNet-val': val_dataset,
            'TinyImageNet-test': test_dataset,
        }

        return
