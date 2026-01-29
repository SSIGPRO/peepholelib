import csv
from pathlib import Path

from PIL import Image

from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.functional.transforms import vgg16_transform

import torch
from torch.utils.data import Dataset

class CheXpertCustom(Dataset):
    def __init__(self, **kwargs):
        Dataset.__init__(self)

        self.path = Path(kwargs["path"])
        self.split = kwargs["split"]
        self.transform = kwargs["transform"]
        self.seed = kwargs["seed"]

        self.csv_path = self.path / f"{self.split}.csv"

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            self.path_column = reader.fieldnames[0]
            self.label_column = reader.fieldnames[-1]
            self.attribute_names = [
                column_name
                for column_name in reader.fieldnames
                if column_name not in {"Study", "Path", "Frontal/Lateral", "AP/PA", "class_label"}
            ]

        self.samples = []
        self._load_samples()

    def _resolve_image_paths(self, raw_path):
        '''
        The image files are stored in train, valid, and test directories, but the CSV may contain either absolute paths or paths relative to these subdirectories.
        This function tries to resolve the image paths accordingly.
        '''
        path_value = str(raw_path).strip()
        if path_value == "":
            return []

        parts = Path(path_value).parts
        relative_path = None
        for split_name in ("train", "valid", "test"):
            if split_name in parts:
                relative_path = Path(*parts[parts.index(split_name) :])
                break

        candidate = self.path / relative_path if relative_path is not None else self.path / path_value

        if candidate.is_file():
            return [candidate]

        if candidate.is_dir():
            return sorted(candidate.glob("*.jpg"))

        return []

    def _load_samples(self):
        missing_images = 0

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_paths = self._resolve_image_paths(row.get(self.path_column, ""))
                if not image_paths:
                    missing_images += 1
                    continue

                class_name = row.get(self.label_column, "")
                if class_name == "":
                    continue

                attributes = torch.tensor(
                    [float(row[attribute_name]) for attribute_name in self.attribute_names],
                    dtype=torch.float32,
                )

                for image_path in image_paths:
                    self.samples.append(
                        (image_path, class_name, attributes)
                    )

        if missing_images > 0:
            print(f"Image file was not found for {missing_images} rows in {self.csv_path}. These were skipped.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_label, attributes = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        sample = {
            "image": img,
            "label": class_label,
        }

        sample.update(
            {
                attribute_name: attributes[i]
                for i, attribute_name in enumerate(self.attribute_names)
            }
        )

        return sample


class CheXpert(DatasetWrap):
    '''
    CheXpert loader for `/srv/newpenny/dataset/CheXpert_clean`.
    Images live in `train/`, `valid/`, and `test/`.
    Labels are stored in `train.csv`, `valid.csv`, and `test.csv`.
    '''

    def __init__(self, **kwargs):
        self.path = Path(kwargs.get("path", "/srv/newpenny/dataset/CheXpert_clean"))
        self.transform = kwargs.get(
            "std_transform", kwargs.get("transform", vgg16_transform)
        )
        self.augmentation = kwargs.get(
            "aug_transform", kwargs.get("augmentation", None)
        )
        self.seed = kwargs.get("seed", 42)

    def __load_data__(self, **kwargs):
        self.__dataset__ = {}

        valid_ds = CheXpertCustom(
            path=self.path,
            split="valid",
            transform=self.transform,
            seed=self.seed,
        )

        train_transform = (
            self.augmentation if self.augmentation is not None else self.transform
        )

        train_ds = CheXpertCustom(
            path=self.path,
            split="train",
            transform=train_transform,
            seed=self.seed,
        )

        test_ds = CheXpertCustom(
            path=self.path,
            split="test",
            transform=self.transform,
            seed=self.seed,
        )
        self.attribute_names = list(valid_ds.attribute_names)

        self.__dataset__ = {
            "CheXpert-train": train_ds,
            "CheXpert-val": valid_ds,
            "CheXpert-test": test_ds,
        }
