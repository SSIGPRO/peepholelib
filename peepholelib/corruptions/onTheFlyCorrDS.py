import warnings

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor
from imagecorruptions import corrupt, get_corruption_names

_NAME_SET = set(get_corruption_names())


class _CorruptedSplit(Dataset):
    def __init__(self, base_dataset, corruption_names, assignments, severity):
        Dataset.__init__(self)
        self.base = base_dataset
        self.corruption_names = corruption_names  # list of str, length K
        self.assignments = assignments            # (N,) long tensor, values in [0, K)
        self.severity = severity

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        raw = self.base[idx]
        if isinstance(raw, dict):
            img_t = raw['image']
            label = raw['label']
        else:
            img_t, label = raw[0], raw[1]

        name = self.corruption_names[self.assignments[idx].item()]
        img_np = np.array(to_pil_image(img_t))
        out = corrupt(img_np, corruption_name=name, severity=self.severity)

        sample = {
            'image':     to_tensor(PILImage.fromarray(out)),
            'label':     label if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long),
            'corruption': self.assignments[idx],
        }
        if isinstance(raw, dict) and 'original_correct' in raw:
            sample['original_correct'] = raw['original_correct']
        return sample


class OnTheFlyCorruptedDS:
    """
    DatasetWrap-compatible class that wraps a clean dataset and applies on-the-fly
    corruptions organised by severity level, matching the structure of CifarC/ImageNetC.

    Each severity level becomes a separate sub-dataset keyed as '{name}-c{level}'.
    Within each level every sample receives exactly one corruption drawn from
    `corruptions`, with equal distribution across samples. The assignment is fixed at
    load time (seeded).

    Args:
        dataset (torch.utils.data.Dataset): Clean base dataset whose samples are dicts
            with at least 'image' (C×H×W float tensor in [0,1]) and 'label', or
            (image, label) tuples.
        corruptions (list[str]): Corruption names to use. Must be a subset of
            imagecorruptions.get_corruption_names().
        name (str): Prefix for __dataset__ keys: '{name}-c{level}'. Default: 'corrupted'.
        severity_levels (list[int]): Severity levels to generate (values in [1, 5]).
            Default: [1, 2, 3, 4, 5].
        seed (int): RNG seed for reproducible corruption assignment. Default: 42.
    """

    has_transforms = False

    def __init__(self, **kwargs):
        self.dataset = kwargs['dataset']
        self.corruption_names = kwargs['corruptions']
        self.name = kwargs.get('name', 'corrupted')
        self.seed = kwargs.get('seed', 42)
        self.severity_levels = kwargs.get('severity_levels', [1, 2, 3, 4, 5])
        self.__dataset__ = {}

        unknown = sorted(set(self.corruption_names) - _NAME_SET)
        if unknown:
            raise ValueError(f'Unknown corruption names: {unknown}. '
                             f'Available: {sorted(_NAME_SET)}')

    def __load_data__(self):
        torch.manual_seed(self.seed)

        N = len(self.dataset)
        K = len(self.corruption_names)
        perm = torch.randperm(N)
        spc = N // K
        rem = N % K

        assignments = torch.empty(N, dtype=torch.long)
        offset = 0
        for ci in range(K):
            count = spc + (1 if ci < rem else 0)
            assignments[perm[offset:offset + count]] = ci
            offset += count

        self.__dataset__ = {}
        for severity in self.severity_levels:
            key = f'{self.name}-c{severity - 1}'
            self.__dataset__[key] = _CorruptedSplit(
                base_dataset=self.dataset,
                corruption_names=self.corruption_names,
                assignments=assignments,
                severity=severity,
            )
