# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap

# torch stuff
import torch
from torch.utils.data import random_split
from torchvision.datasets import INaturalist
from torchvision.transforms import ToTensor, Compose, Resize


class iNaturalistCustom(INaturalist):

    def __init__(self, **kwargs):
        INaturalist.__init__(self, **kwargs)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return {
            'image': img,
            'label': torch.tensor(label),
        }


class iNaturalist(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        iNaturalist Far-OOD loader (val & test). Used as a Far-OOD dataset in the OpenOOD benchmark.
        containing all images).

        Args:
            path (str): INaturalist download folder.
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
        _data = iNaturalistCustom(
            root = self.path,
            version = '2021_train_mini',
            transform = self.transform,
            download = True,
        )

        val_dataset, test_dataset = random_split(
            _data,
            self.splitting_ratio,
            generator=torch.Generator().manual_seed(self.seed)
        )
        print(len(val_dataset), len(test_dataset))

        self.__dataset__ = {
            'iNaturalist-val': val_dataset,
            'iNaturalist-test': test_dataset,
        }

        return
