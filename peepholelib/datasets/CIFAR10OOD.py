# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap

# torch stuff
import torch
from torch.utils.data import random_split

# CIFAR from torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose


class CIFAR10OODCustom(CIFAR10):
    def __init__(self, **kwargs):
        CIFAR10.__init__(self, **kwargs)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return {
            'image': img,
            'label': torch.tensor(label),
        }


class CIFAR10OOD(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        CIFAR-10 Near-OOD loader (val & test). Used as a Near-OOD dataset in the OpenOOD benchmark for CIFAR100.
        The CIFAR-10 test split (10000 samples) is divided into val and test according to `splitting_ratio`.

        Args:
            path (str): CIFAR-10 download folder. Downloaded automatically if not present.
            transform (callable, optional): Transform applied to images. Defaults to ToTensor.
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
            self.transform = Compose([ToTensor()])

        return

    def __load_data__(self):
        _data = CIFAR10OODCustom(
            root=self.path,
            train=False,
            transform=self.transform,
            download=True,
        )

        val_dataset, test_dataset = random_split(
            _data,
            self.splitting_ratio,
            generator=torch.Generator().manual_seed(self.seed)
        )

        self.__dataset__ = {
            'CIFAR10OOD-val': val_dataset,
            'CIFAR10OOD-test': test_dataset,
        }

        return
