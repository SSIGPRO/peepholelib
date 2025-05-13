# ---------------------------------------------------------------------
# ImageNet‑1K loader (train + val).
# Expected directory passed via data_path:
#   .../imagenet-1k/data/{train,val}/<class>/*.JPEG
#
# NOTE – there is no test split because ImageNet test labels are private.
# TODO – consider applying light augmentation to the val split (open issue).
# ---------------------------------------------------------------------

# peepholelib imports
from .dataset_base import DatasetBase
from .transforms import vgg16_imagenet, vgg16_imagenet_augmentations

from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

class ImageNet(DatasetBase):
    """
    ImageNet‑1K loader (train & val).
    Expects:
        <data_path>/train/<class>/*.JPEG
        <data_path>/val/<class>/*.JPEG
    Returns from load_data():
        {'train': DataLoader, 'val': DataLoader}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = "imagenet-1k"
        print(f"dataset: {self.dataset}")

    def load_data(self, *, batch_size: int, seed: int,
                  transform=None, augmentation_transform=None) -> Dict[str, DataLoader]:

        transform = transform or vgg16_imagenet
        augmentation_transform = (
            augmentation_transform if augmentation_transform is not None else transform
        )

        root = Path(self.data_path).expanduser().resolve()
        train_dir, val_dir = root / "train", root / "val"
        if not train_dir.is_dir():
            raise FileNotFoundError(f"Missing {train_dir}")

        torch.manual_seed(seed)

        # datasets
        train_ds = datasets.ImageFolder(train_dir, transform=augmentation_transform)
        val_ds   = datasets.ImageFolder(val_dir,   transform=transform)

        # dataloaders
        self._dls = {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=8, pin_memory=True),
            "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                num_workers=8, pin_memory=True),
        }

        # metadata
        self._dss = {"train": train_ds, "val": val_ds}
        self._classes = {i: c for i, c in enumerate(train_ds.classes)}

        return self._dls