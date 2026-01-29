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
from torch.utils.data import Subset, random_split

# peepholelib imports
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.functional.transforms import vgg16 as transform

class ImageNetCustom(IN1K):

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        sample = {
            "image": img,
            "label": torch.tensor(label),
        }
        return sample

class ImageNet(DatasetWrap):
    def __init__(self, **kwargs):
        """
        ImageNet-1K loader (train & val & test). A train/val split is created
        from the official ImageNet training split using `train_ratio`, while the
        official validation split is exposed as `ImageNet-test`.

        Args:
            path (str): Path to the ImageNet-1K root folder containing `train/`
                and `val/` subfolders.
            transform (callable, optional): Transform applied to validation/test
                images. Defaults to `vgg16`.
            augmentation (callable, optional): If provided, applied only to the
                training split.
            train_ratio (float, optional): Fraction of official training samples
                used for train (remainder goes to val).
            seed (int, optional): Random seed used for deterministic train/val
                splitting.
        Returns:
            - a thumbs up
        """

        # add a default transform for specific DS
        if 'transform' not in kwargs:
            kwargs['transform'] = transform

        self.augmentation = kwargs.get('augmentation', None)
        self.train_ratio = kwargs.get('train_ratio', 0.8)

        DatasetWrap.__init__(self, **kwargs)

        return

    def __load_data__(self, **kwargs):
        '''
        Load and prepare ImageNet data.
        
        Returns:
        - a thumbs up
        '''

        transform = self.transform
        augmentation = self.augmentation
        seed = self.seed

        test_ds = ImageNetCustom(
                root=self.path,
                split='val',
                transform=transform
            )
        
        base_ds = ImageNetCustom(
                root=self.path,
                split='train',
                transform=transform
            )
        
        train_idx, val_idx = random_split(
                range(len(base_ds)),
                [self.train_ratio, 1 - self.train_ratio],
                generator=torch.Generator().manual_seed(seed)
            )
        
        val_ds = Subset(base_ds, val_idx)

        if self.augmentation is None:

            train_ds = Subset(base_ds, train_idx)
            
        else:
            _train_aug = ImageNetCustom(
                root=self.path,
                split='train',
                transform=augmentation,
                download=True
            )
            train_ds = Subset(_train_aug, train_idx)

        
        self.__dataset__ = {
                "ImageNet-train": train_ds,
                "ImageNet-val": val_ds,
                "ImageNet-test": test_ds
            }

        return
    
    @classmethod
    def get_classes(cls, **kwargs):
        root = kwargs['path']
        meta_path = Path(root) / 'meta.bin'

        # torchvision ImageNet meta.bin stores wnid -> tuple of class names.
        meta = torch.load(meta_path, map_location='cpu')
        if isinstance(meta, dict) and 'wnid_to_classes' in meta:
            wnid_to_classes = meta['wnid_to_classes']
        elif isinstance(meta, tuple) and len(meta) > 0 and isinstance(meta[0], dict):
            wnid_to_classes = meta[0]
        else:
            raise ValueError(f"Unsupported ImageNet meta format in '{meta_path}'.")

        train_dir = root / 'train'

        # Keep same class order used by torchvision Folder datasets (sorted folder names).
        wnids = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

        labels = {}
        for idx, wnid in enumerate(wnids):
            names = wnid_to_classes.get(wnid, (wnid,))
            labels[idx] = names[0] if isinstance(names, (tuple, list)) else str(names)

        return labels
