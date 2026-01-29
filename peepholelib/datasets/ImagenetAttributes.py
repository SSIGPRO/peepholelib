import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor

from peepholelib.datasets.datasetWrap import DatasetWrap


class ImageNetAttrCustom(Dataset):
    ILSVRC_ROOT = Path("/srv/newpenny/dataset/ImagenetAttributes/ILSVRC")
    DATA_ROOT = ILSVRC_ROOT / "Data" / "DET"
    IMAGE_SETS_ROOT = ILSVRC_ROOT / "ImageSets" / "DET"

    TRAIN_ANNOTATIONS = Path(
        "/srv/newpenny/dataset/ImagenetAttributes/imagenet_attributes_train.jsonl"
    )
    VAL_ANNOTATIONS = Path(
        "/srv/newpenny/dataset/ImagenetAttributes/imagenet_attributes_val.jsonl"
    )

    _annotation_cache = {}
    _class_cache = {}
    _image_label_cache = {}

    def __init__(self, split, transform=None):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'.")

        self.split = split
        self.transform = transform
        self.annotations_path = None
        if split == "train":
            self.annotations_path = self.TRAIN_ANNOTATIONS
        elif split == "val":
            self.annotations_path = self.VAL_ANNOTATIONS

        self.annotation_by_stem = self._load_annotations(self.annotations_path)
        self.image_to_label = self._load_image_to_label(self.annotations_path)
        self.class_to_idx = self._load_class_to_idx()
        self.samples, self.filtered_annotations = self._build_samples()
        self.imgs = self.samples
        self.targets = [label for _, label in self.samples]

    @classmethod
    def _image_path_for_sample(cls, split, sample_id):
        if split == "train":
            return cls.DATA_ROOT / "train" / f"{sample_id}.JPEG"
        return cls.DATA_ROOT / split / f"{sample_id}.JPEG"

    def _build_samples(self):
        samples = []
        filtered_annotations = []
        split_file = self.IMAGE_SETS_ROOT / f"{self.split}.txt"

        with open(split_file, "r") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue

                sample_id = parts[0]
                image_path = self._image_path_for_sample(self.split, sample_id)
                if not image_path.exists():
                    continue

                stem = image_path.stem
                annotations = self.annotation_by_stem.get(stem)
                if self.split != "test" and annotations is None:
                    continue

                label = -1
                if self.split != "test":
                    raw_label = self.image_to_label.get(stem)
                    label = self.class_to_idx.get(raw_label, -1)
                    if label < 0:
                        continue

                samples.append((str(image_path), label))
                filtered_annotations.append(annotations or [])

        return samples, filtered_annotations

    @classmethod
    def _load_annotations(cls, annotations_path):
        if annotations_path is None:
            return {}

        cache_key = str(annotations_path)
        if cache_key in cls._annotation_cache:
            return cls._annotation_cache[cache_key]

        annotation_by_stem = defaultdict(list)
        with open(annotations_path, "r") as handle:
            for line in handle:
                row = json.loads(line)
                stem = row["belong_to_im"]
                annotation_by_stem[stem].append(
                    {
                        "annotation_index": int(
                            row.get("annotation_index", len(annotation_by_stem[stem]))
                        ),
                        "attribute": cls._to_float_list(row.get("attribute", [])),
                        "specific_attribute": cls._to_float_list(
                            row.get("specific_attribute", [])
                        ),
                    }
                )

        normalized_annotations = {}
        for stem, annotations in annotation_by_stem.items():
            annotations.sort(key=lambda item: item["annotation_index"])
            normalized_annotations[stem] = annotations

        cls._annotation_cache[cache_key] = normalized_annotations
        return normalized_annotations

    @staticmethod
    def _parse_raw_label(row):
        class_id = row.get("class_id")
        if class_id is not None:
            return int(class_id)

        class_dir = str(row.get("class_dir", ""))
        match = re.match(r"(?:cls_)?(\d+)", class_dir)
        if match is None:
            raise RuntimeError(f"Could not parse ImageNetAttr class label from '{class_dir}'.")
        return int(match.group(1))

    @classmethod
    def _load_image_to_label(cls, annotations_path):
        if annotations_path is None:
            return {}

        cache_key = str(annotations_path)
        if cache_key in cls._image_label_cache:
            return cls._image_label_cache[cache_key]

        per_image_counts = defaultdict(Counter)
        with open(annotations_path, "r") as handle:
            for line in handle:
                row = json.loads(line)
                stem = row["belong_to_im"]
                per_image_counts[stem][cls._parse_raw_label(row)] += 1

        image_to_label = {}
        for stem, counts in per_image_counts.items():
            best_count = max(counts.values())
            winners = [label for label, count in counts.items() if count == best_count]
            image_to_label[stem] = min(winners)

        cls._image_label_cache[cache_key] = image_to_label
        return image_to_label

    @classmethod
    def _load_class_to_idx(cls):
        cache_key = f"{cls.TRAIN_ANNOTATIONS}:{cls.VAL_ANNOTATIONS}"
        if cache_key in cls._class_cache:
            return cls._class_cache[cache_key]

        labels = set(cls._load_image_to_label(cls.TRAIN_ANNOTATIONS).values())
        labels.update(cls._load_image_to_label(cls.VAL_ANNOTATIONS).values())
        class_to_idx = {label: idx for idx, label in enumerate(sorted(labels))}
        cls._class_cache[cache_key] = class_to_idx
        return class_to_idx

    @staticmethod
    def _to_float_list(values):
        if values is None:
            return []
        if isinstance(values, (int, float)):
            return [float(values)]
        return [float(value) for value in values]

    @staticmethod
    def _aggregate_annotations(values_per_annotation, fixed_size=None):
        if fixed_size is None:
            max_len = max((len(values) for values in values_per_annotation), default=0)
        else:
            max_len = fixed_size

        aggregated = torch.zeros(max_len, dtype=torch.float32)
        for values in values_per_annotation:
            row = torch.zeros(max_len, dtype=torch.float32)
            if max_len > 0:
                limit = min(len(values), max_len)
                if limit > 0:
                    row[:limit] = torch.tensor(values[:limit], dtype=torch.float32)
            aggregated = torch.maximum(aggregated, row)

        return aggregated

    def __getitem__(self, index):
        sample_path, label = self.samples[index]
        img = default_loader(sample_path)
        if self.transform is not None:
            img = self.transform(img)
        if not torch.is_tensor(img):
            img = to_tensor(img)

        annotations = self.filtered_annotations[index]
        sample = {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
            "specific_attribute": self._aggregate_annotations(
                [annotation["specific_attribute"] for annotation in annotations],
                fixed_size=10,
            ),
            "attribute": self._aggregate_annotations(
                [annotation["attribute"] for annotation in annotations]
            ),
        }
        return sample

    def __len__(self):
        return len(self.samples)


class ImageNetAttr(DatasetWrap):
    def __init__(self, **kwargs):
        self.transform = kwargs.get("std_transform")
        self.augmentation = kwargs.get("aug_transform", None)

        DatasetWrap.__init__(self, **kwargs)

    def __load_data__(self, **kwargs):
        train_transform = self.augmentation or self.transform

        train_ds = ImageNetAttrCustom(
            split="train",
            transform=train_transform,
        )

        val_ds = ImageNetAttrCustom(
            split="val",
            transform=self.transform,
        )

        self.__dataset__ = {
            "ImageNetAttr-train": train_ds,
            "ImageNetAttr-val": val_ds,
        }

    @classmethod
    def get_classes(cls, **kwargs):
        class_to_idx = ImageNetAttrCustom._load_class_to_idx()
        return {idx: label for label, idx in class_to_idx.items()}
