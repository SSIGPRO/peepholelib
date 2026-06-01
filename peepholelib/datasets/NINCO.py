# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap

# torch stuff
import torch
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize


class NINCOCustom(ImageFolder):

    def __init__(self, **kwargs):
        ImageFolder.__init__(self, **kwargs)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return {
            'image': img,
            'label': torch.tensor(label),
        }


class NINCO(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        NINCO Near-OOD loader (val & test). Used as a Near-OOD dataset in the OpenOOD benchmark.
        Data is expected in ImageFolder format (one subdirectory per class, or a single subdirectory
        containing all images).

        Args:
            path (str): Path to the NINCO folder (ImageFolder-compatible layout).
            transform (callable, optional): Transform applied to images. Defaults to Resize(224) + ToTensor.
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
            self.transform = Compose([ToTensor(), Resize((224, 224))])

        return

    def __load_data__(self):
        _data = NINCOCustom(
            root=self.path,
            transform=self.transform,
        )

        val_dataset, test_dataset = random_split(
            _data,
            self.splitting_ratio,
            generator=torch.Generator().manual_seed(self.seed)
        )

        self.__dataset__ = {
            'NINCO-val': val_dataset,
            'NINCO-test': test_dataset,
        }

        return
