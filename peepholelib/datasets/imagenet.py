# Our stuff
from .dataset_base import DatasetBase
## from .transforms import vgg16_ImageNet

# General python stuff
from pathlib import Path as Path

# torch stuff
import torch
from torch.utils.data import random_split, DataLoader

# ImageNet from torchvision
from torchvision import transforms, datasets


class ImageNet(DatasetBase):
    def __init__(self, **kwargs):
        DatasetBase.__init__(self, **kwargs)

        # use ImageNet by default
        if 'dataset' in kwargs:
            self.dataset = kwargs['dataset']
        else:
            self.dataset = 'ImageNet'
        print('dataset: %s' % self.dataset)

        # raise error if the dataset is not ImageNet
        if "imagenet" not in self.dataset.lower():
            raise ValueError("Dataset must be ImageNet")

        '''
        ImageNet-1k num_classes: 1e3
        '''
        return

    def load_data(self, **kwargs):
        '''
        Load and prepare data for a specified portion of a dataset.

        Args:
        - dataset (str): The name of the dataset ('ImageNet').
        - batch_size (int): The batch size for DataLoader.
        - seed (int): Random seed for reproducibility.
        - data_augmentation (bool): Flag indicating whether to apply data
        augmentation (default: False).
        - original_transform (torchvision.transforms.Compose): Custom transform to apply to the original dataset. (default: ImageNet transform)
        - augmentation_transform (torchvision.transforms.Compose): Custom transform to apply to the augmented dataset. (default: ImageNet transform)

        Returns:
        - dict: containing a DataLoader for 'train', 'val', 'test', and a dictionary mapping class indices to class names for 'classes'.

        Example:
        - To load the training data of ImageNet with a batch size of 32:
        >>> c = ImageNet(dataset = 'ImageNet')
        >>> loaders = c.load_data(batch_size=32, seed=42)
        '''

        # parse parameteres
        batch_size = kwargs['batch_size']
        seed = kwargs['seed']

        augmentation_transform = kwargs['augmentation_transform'] if 'augmentation_transform' in kwargs else None

        # original dataset without augmentation
        # accepts custom transform if provided in kwargs
        transform = kwargs['transform'] if 'transform' in kwargs else vgg16_ImageNet

        # set torch seed
        torch.manual_seed(seed)

        # Load the official validation split as the test dataset.
        test_dataset = datasets.ImageNet(
            root=self.data_path,
            split='val',
            transform=transform,
            download=False
        )
        
        # Load the training split without any transform yet.
        _train_data = datasets.ImageNet(
            root=self.data_path,
            split='train',
            transform=None,
            download=False
        )

        train_dataset, val_dataset = random_split(
            _train_data,
            [0.8, 0.2],
            generator=torch.Generator().manual_seed(seed)
        )

        # set validation dataset transform
        val_dataset.dataset.transform = transform

        # Apply the transformation according to data augmentation
        if augmentation_transform != None:
            train_dataset.dataset.transform = augmentation_transform
        else:
            train_dataset.dataset.transform = transform

        # Save datasets as objects in the class

        self._dss = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
                }
        self._classes = {i: class_name for i, class_name in enumerate(test_dataset.classes)}

        return
