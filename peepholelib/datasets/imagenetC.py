from collections import defaultdict
from math import floor

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor 

from peepholelib.datasets.datasetWrap import DatasetWrap

class CustomDS(Dataset):
    def __init__(self, samples, synset_to_label):
        Dataset.__init__(self)
        self._to_tensor = ToTensor()

        p = [
            "brightness",
            "contrast",
            "defocus_blur",
            "elastic_transform",
            "fog",
            "frost",
            "gaussian_blur",
            "gaussian_noise",
            "glass_blur",
            "impulse_noise",
            "jpeg_compression",
            "motion_blur",
            "pixelate",
            "saturate",
            "shot_noise",
            "snow",
            "spatter",
            "speckle_noise",
            "zoom_blur",
        ]
        self.mapping = {c: i for i, c in enumerate(p)}
        self.samples = samples
        self.synset_to_label = synset_to_label
        unknown = sorted({corruption for _, _, corruption in samples} - set(self.mapping.keys()))
        if unknown:
            raise ValueError(f"Unknown corruption names found: {unknown}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, synset, corruption = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(self.synset_to_label[synset], dtype=torch.long)
        return {
            "image": self._to_tensor(image),
            "label": label,
            "corruption": torch.tensor(self.mapping[corruption], dtype=torch.long),
        }

class ImageNetC(DatasetWrap):
    def __init__(self, **kwargs):
        """
        ImageNet-C loader (val & test). Builds corruption-level datasets from the ImageNet-C directory hierarchy.

        Args:
            path (str): Path to the ImageNet-C root folder organized as `<corruption>/<severity>/<synset>/*.JPEG`.
            seed (int, optional): Random seed used for reproducible corruption mixing.
            corruptions (list[str], optional): Optional list of corruptions to consider.
        Returns:
            - a thumbs up
        """
        self.corruptions = kwargs.get("corruptions", None)
        DatasetWrap.__init__(self, **kwargs)

    def __load_data__(self):
        """
        Load and prepare ImageNet-C data.
        """
        torch.manual_seed(self.seed)

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: '{self.path}'.")
        if not self.path.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: '{self.path}'.")

        # samples_by_severity_and_corruption[severity][corruption] -> [(image_path, synset, corruption), ...]
        samples_by_severity_and_corruption = defaultdict(lambda: defaultdict(list))
        synsets = set()

        for corruption_dir in sorted(p for p in self.path.iterdir() if p.is_dir()):
            corruption = corruption_dir.name

            for severity_dir in sorted(p for p in corruption_dir.iterdir() if p.is_dir()):
                severity = int(severity_dir.name)

                for class_dir in sorted(p for p in severity_dir.iterdir() if p.is_dir()):
                    synset = class_dir.name
                    for image_path in sorted(p for p in class_dir.iterdir() if p.is_file()):
                        samples_by_severity_and_corruption[severity][corruption].append(
                            (str(image_path), synset, corruption)
                        )
                        synsets.add(synset)

        synset_to_label = {synset: i for i, synset in enumerate(sorted(synsets))}

        self.__dataset__ = {}
        for severity in range(1, 6):
            corruption_to_samples = samples_by_severity_and_corruption.get(severity, {})
            if not corruption_to_samples:
                continue

            corruptions = sorted(corruption_to_samples.keys())
            n_corruptions = len(corruptions)
            n_samples = min(len(corruption_to_samples[c]) for c in corruptions)
            if n_samples == 0:
                continue

            idxs = torch.randperm(n_samples).tolist()
            spc = floor(n_samples / n_corruptions)
            mixed_test = [None] * n_samples
            mixed_val = [None] * n_samples

            corruptions_val = list(reversed(corruptions))
            last_ci = -1
            for ci, (corr_test, corr_val) in enumerate(zip(corruptions, corruptions_val)):
                dst_idxs = idxs[ci * spc:(ci + 1) * spc]
                src_test = corruption_to_samples[corr_test]
                src_val = corruption_to_samples[corr_val]
                for di in dst_idxs:
                    mixed_test[di] = src_test[di]
                    mixed_val[di] = src_val[di]
                last_ci = ci

            if last_ci >= 0:
                rem_idxs = idxs[(last_ci + 1) * spc:]
                src_test = corruption_to_samples[corruptions[last_ci]]
                src_val = corruption_to_samples[corruptions_val[last_ci]]
                for di in rem_idxs:
                    mixed_test[di] = src_test[di]
                    mixed_val[di] = src_val[di]

            ds_test = CustomDS(
                samples=mixed_test,
                synset_to_label=synset_to_label,
            )
            ds_val = CustomDS(
                samples=mixed_val,
                synset_to_label=synset_to_label,
            )

            c_idx = severity - 1
            self.__dataset__[f"ImageNet-C-val-c{c_idx}"] = ds_val
            self.__dataset__[f"ImageNet-C-test-c{c_idx}"] = ds_test

    def get(self, ds_key, idx):
        if not self.__dataset__:
            raise RuntimeError("Data not loaded. Please run load_data() first.")
        return [self.__dataset__[ds_key][idx]]
