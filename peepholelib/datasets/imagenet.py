# ---------------------------------------------------------------------
# ImageNet‑1K loader (train + val).
# Expected directory passed via data_path:
#   .../imagenet-1k/data/{train,val}/<class>/*.JPEG
#
# NOTE – there is no test split because ImageNet test labels are private.
# TODO – consider applying light augmentation to the val split (open issue).
# ---------------------------------------------------------------------

# general python stuff
from pathlib import Path
import pickle

# torch stuff
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet as IN1K
from torch.utils.data import random_split

# peepholelib imports
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.functional.transforms import vgg16_imagenet

class ImageNetCustom(IN1K):

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        return torch.tensor(img), torch.tensor(target)

class ImageNet(DatasetWrap):
    def __init__(self, **kwargs):
        """
        ImageNet‑1K loader (train & val).
        Expects:
            path (str): imagenet download folder
        Returns:
            - a thumbs up
        """

        # add a default transform for specific DS
        if 'transform' not in kwargs:
            kwargs['transform'] = vgg16_imagenet

        augmentation = kwargs.get('augmentation', None)

        DatasetWrap.__init__(self, **kwargs)

        return

    def __load_data__(self, **kwargs):
        '''
        Load and prepare Imagenet data.
        
        Returns:
        - a thumbs up
        '''

        transform = self.transform
        seed = self.seed

        # set torch seed
        torch.manual_seed(seed)

        test_ds = ImageNetCustom(
                root=self.path,
                split='val',
                transform=transform
            )

        if self.augmentation is None:

            full_train = ImageNetCustom(
                root=self.path,
                split='train',
                transform=transform
            )

            train_ds, val_ds = random_split(
                full_train,
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(self.seed)
            )
            
        else:
            full_train_aug = ImageNetCustom(
                root=self.path,
                split='train',
                transform=self.augmentation
            )

            full_train_noaug = ImageNetCustom(
                root=self.path,
                split='train',
                transform=transform
            )

            g = torch.Generator().manual_seed(seed)
            train_idx, val_idx = random_split(
                range(len(full_train_aug)),
                [0.8, 0.2],
                generator=g
            )
            train_ds = torch.utils.data.Subset(full_train_aug, train_idx)
            val_ds = torch.utils.data.Subset(full_train_noaug, val_idx)

        
        self.__dataset__ = {
                "ImageNet-train": train_ds,
                "ImageNet-val": val_ds,
                "ImageNet-test": test_ds
            }

        self._classes = {i: c for i, c in enumerate(train_ds.classes)}

        return
    
    ## TODO: test this method
    @classmethod
    def get_classes(cls, **kwargs):
        meta_path = kwargs['meta_path']
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        labels = {i: name for i, name in enumerate(meta['fine_label_names'])}
        return labels 
    
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
