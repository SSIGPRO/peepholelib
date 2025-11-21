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

# peepholelib imports
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.transforms import vgg16_imagenet

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

        # datasets
        train_ds = IN1K(
                root = self.path,
                split = 'train',
                transform=transform
                )
        val_ds = IN1K(
                root = self.path,
                split = 'val',
                transform=transform
                )

        # metadata
        self.__dataset__ = {"ImageNet-train": train_ds, "ImageNet-val": val_ds}
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
