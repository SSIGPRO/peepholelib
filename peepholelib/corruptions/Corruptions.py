import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.transforms.functional import to_pil_image, to_tensor
from imagecorruptions import corrupt, get_corruption_names

from peepholelib.corruptions.corruption_base import CorruptionBase

ALL_CORRUPTION_NAMES = get_corruption_names()
_NAME_SET = set(ALL_CORRUPTION_NAMES)


class RandomCorruption(CorruptionBase):
    """
    On-the-fly corruption perturbation with equal distribution across corruption types.

    Args:
        severity (int): Corruption severity in [1, 5]. Default: 1.
        corruptions (list[str], optional): Names of corruptions to use. Defaults to all
            19 Hendrycks corruptions. Call get_corruption_names() for the full list.
        seed (int, optional): RNG seed for reproducible corruption assignment per batch.

    Shape:
        - images: (N, C, H, W) float tensor in [0, 1].
        - output images: (N, C, H, W) float tensor in [0, 1].
        - output corruption_ids: (N,) long tensor, index into self.corruption_names.
    """

    def __init__(self, **kwargs):
        CorruptionBase.__init__(self, **kwargs)
        self.seed = kwargs.get('seed', None)

        requested = kwargs.get('corruptions', None)
        if requested is not None:
            unknown = sorted(set(requested) - _NAME_SET)
            if unknown:
                raise ValueError(f'Unknown corruption names: {unknown}. '
                                 f'Available: {sorted(_NAME_SET)}')
            self._corruption_names = list(requested)
        else:
            self._corruption_names = list(ALL_CORRUPTION_NAMES)

        self._rng = torch.Generator()
        if self.seed is not None:
            self._rng.manual_seed(self.seed)

    @property
    def corruption_names(self):
        return self._corruption_names

    def __call__(self, images, labels):
        """
        Args:
            images: (N, C, H, W) float tensor in [0, 1].
            labels: (N,) long tensor (unused, kept for interface parity with attacks).

        Returns:
            corrupted_images: (N, C, H, W) float tensor in [0, 1].
            corruption_ids:   (N,) long tensor — index into self.corruption_names per sample.
        """
        N = images.shape[0]
        K = len(self._corruption_names)

        perm = torch.randperm(N, generator=self._rng).tolist()
        spc = N // K
        rem = N % K

        result = torch.empty_like(images)
        corruption_ids = torch.empty(N, dtype=torch.long)
        offset = 0
        for ci, name in enumerate(self._corruption_names):
            count = spc + (1 if ci < rem else 0)
            idxs = perm[offset:offset + count]
            offset += count
            for idx in idxs:
                img_np = np.array(to_pil_image(images[idx]))
                out = corrupt(img_np, corruption_name=name, severity=self.severity)
                result[idx] = to_tensor(PILImage.fromarray(out))
                corruption_ids[idx] = ci

        return result, corruption_ids
