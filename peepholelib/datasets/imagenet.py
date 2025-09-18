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

# torch stuff
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet as IN1K

# peepholelib imports
from peepholelib.datasets.dataset_base import DatasetBase
from peepholelib.datasets.transforms import vgg16_imagenet

class ImageNet(DatasetBase):
    """
    ImageNet‑1K loader (train & val).
    Expects:
        data_path (str): imagenet download folder
    Returns:
        - a thumbs up
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = "imagenet-1k"
        print(f"dataset: {self.dataset}")
        return

    def load_data(self, **kwargs):
        '''
        Load and prepare Imagenet data.
        
        Args:
        - seed (int): Random seed for reproducibility.
        - transform (torchvision.transforms.Compose): Custom transform to apply to the original dataset. (default: ImageNet1k for vgg16 transform)
        
        Returns:
        - a thumbs up
        '''

        transform = kwargs['transform'] if 'transform' in kwargs else vgg16_imagenet

        seed = kwargs['seed']
        torch.manual_seed(seed)

        # datasets
        train_ds = IN1K(
                root = self.data_path,
                split = 'train',
                transform=transform
                )
        val_ds = IN1K(
                root = self.data_path,
                split = 'val',
                transform=transform
                )

        # metadata
        self.__dataset__ = {"ImageNet-train": train_ds, "ImageNet-val": val_ds}
        self._classes = {i: c for i, c in enumerate(train_ds.classes)}

        return
    
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
