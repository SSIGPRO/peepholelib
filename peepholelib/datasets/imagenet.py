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
from torchvision.datasets import ImageNet as IN1K
from torch.utils.data import Subset, random_split
from torchvision.transforms import ToTensor, Compose, Resize 

# peepholelib imports
from peepholelib.datasets.datasetWrap import DatasetWrap

class ImageNetCustom(IN1K):
    def __init__(self, **kwargs):
        IN1K.__init__(self, **kwargs)
        # each image in ImageNet has a differente size and resolution
        self._to_tensor = Compose([ToTensor(), Resize((224, 224))])

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        sample = {
            "image": self._to_tensor(img),
            "label": torch.tensor(label),
        }
        return sample

class ImageNet(DatasetWrap):
    def __init__(self, **kwargs):
        """
        ImageNet-1K loader (train & val & test). A train/val split is created the official ImageNet training split using `train_ratio`, while the official validation split is exposed as `ImageNet-test`.

        Args:
            path (str): Path to the ImageNet-1K root folder containing `train/` and `val/` subfolders.
            augmentation (callable, optional): If provided, applied only to the training split.
            train_ratio (float, optional): Fraction of official training samples used for train (remainder goes to val).
            seed (int, optional): Random seed used for deterministic train/val splitting.
        Returns:
            - a thumbs up
        """
        self.train_ratio = kwargs.get('train_ratio', 0.8)

        DatasetWrap.__init__(self, **kwargs)

        return

    def __load_data__(self, **kwargs):
        '''
        Load and prepare ImageNet data.
        
        Returns:
        - a thumbs up
        '''

        test_ds = ImageNetCustom(
                root=self.path,
                split='val',
            )
        
        base_ds = ImageNetCustom(
                root=self.path,
                split='train',
            )
        
        train_ds, val_ds = random_split(
                base_ds,
                [self.train_ratio, 1 - self.train_ratio],
                generator=torch.Generator().manual_seed(self.seed)
            )
        
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
