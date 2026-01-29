# general python stuff
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.functional.transforms import vgg16_transform

# torch stuff
import torch
from torch.utils.data import Dataset, Subset

def _iter_data_lines(file_path):
    """
    DeepFashion mapping files have:
    line 1 = count, line 2 = header, then data.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        f.readline()
        f.readline()
        for line in f:
            line = line.strip()
            if line:
                yield line


class CustomDS(Dataset):
    def __init__(self, **kwargs):
        """
        Unified DeepFashion loader using official coarse files:
        - list_eval_partition.txt
        - list_category_img.txt
        - list_attr_img.txt
        """
        Dataset.__init__(self)

        self.path = Path(kwargs["path"])
        self.transform = kwargs["transform"]
        self.seed = kwargs["seed"]
        self.top_k_attributes = kwargs.get("top_k_attributes", None)
        self.del_no_attributes_samples = bool(
            kwargs.get("del_no_attributes_samples", False)
        )

        if self.top_k_attributes is not None:
            self.top_k_attributes = int(self.top_k_attributes)
            if self.top_k_attributes < 0:
                raise ValueError("top_k_attributes must be >= 0 or None.")

        # ---- required files ----
        category_file = self.path / "list_category_cloth.txt"
        attr_vocab_file = self.path / "list_attr_cloth.txt"
        coarse_partition_file = self.path / "list_eval_partition.txt"
        coarse_category_img_file = self.path / "list_category_img.txt"
        coarse_attr_img_file = self.path / "list_attr_img.txt"

        # ---- classes ----
        self.id_to_class = {}
        raw_category_id = 1
        for line in _iter_data_lines(category_file):
            match = re.search(r"\s+(\d+)\s*$", line)
            if match is None:
                continue
            category_name = line[: match.start()].strip()
            self.id_to_class[raw_category_id - 1] = category_name
            raw_category_id += 1

        # ---- attribute vocabulary ----
        self.attribute_names = []
        for line in _iter_data_lines(attr_vocab_file):
            match = re.search(r"\s+(\d+)\s*$", line)
            if match is None:
                continue
            attr_name = line[: match.start()].strip()
            self.attribute_names.append(attr_name)

        self._all_attribute_names = list(self.attribute_names)
        self._n_all_attributes = len(self._all_attribute_names)
        self._all_attribute_keys = list(self._all_attribute_names)

        # ---- optional bounding boxes ----
        self.image_to_bbox = {}
        bbox_file = self.path / "list_bbox.txt"
        if bbox_file.exists():
            for line in _iter_data_lines(bbox_file):
                tokens = line.split()
                if len(tokens) < 5:
                    continue
                image_name = tokens[0]
                try:
                    x1, y1, x2, y2 = [float(v) for v in tokens[1:5]]
                except ValueError:
                    continue
                self.image_to_bbox[image_name] = torch.tensor(
                    [x1, y1, x2, y2], dtype=torch.float32
                )

        self.samples = []  # (image_name, class_id, raw_attributes)
        self.sample_splits = []  # "train" | "val" | "test"
        self._load_coarse_splits(
            partition_file=coarse_partition_file,
            category_img_file=coarse_category_img_file,
            attr_img_file=coarse_attr_img_file,
        )
        self._configure_attribute_subset()
        if self.del_no_attributes_samples:
            self._drop_samples_without_attributes()

    def _configure_attribute_subset(self):
        '''
        Configures the subset of attributes to return based on top_k_attributes.
        If top_k_attributes is None, all attributes are kept. Otherwise, only the k most frequent attributes are kept.
        '''
        if self.top_k_attributes is not None:
            if self.top_k_attributes == 0:
                selected = []
            else:
                popularity = torch.zeros(self._n_all_attributes, dtype=torch.int64)
                for _, _, raw_attributes in self.samples:
                    popularity += (raw_attributes > 0).to(torch.int64)
                ranked = sorted(
                    range(self._n_all_attributes),
                    key=lambda idx: (-int(popularity[idx]), idx),
                )
                selected = ranked[: self.top_k_attributes]
        else:
            selected = list(range(self._n_all_attributes))

        self._attribute_indices = tuple(selected)
        self.attribute_names = [
            self._all_attribute_names[idx] for idx in self._attribute_indices
        ]
        self.attribute_keys = [
            self._all_attribute_keys[idx] for idx in self._attribute_indices
        ]
        self.n_attributes = len(self._attribute_indices)

    def _select_attributes(self, raw_attributes):
        '''
        Selects a subset of attributes from raw_attributes based on top_k_attributes configuration.
        '''
        if self.top_k_attributes is None:
            return raw_attributes
        if len(self._attribute_indices) == 0:
            return raw_attributes[:0]
        index_tensor = torch.tensor(self._attribute_indices, dtype=torch.long)
        return raw_attributes.index_select(0, index_tensor)

    def _drop_samples_without_attributes(self):
        kept_samples = []
        kept_splits = []
        for sample, split_name in zip(self.samples, self.sample_splits):
            _, _, raw_attributes = sample
            if torch.any(self._select_attributes(raw_attributes) > 0):
                kept_samples.append(sample)
                kept_splits.append(split_name)
        self.samples = kept_samples
        self.sample_splits = kept_splits

    def _load_coarse_splits(self, partition_file, category_img_file, attr_img_file):
        image_to_label, image_to_attrs = self._load_coarse_annotations(
            category_img_file=category_img_file,
            attr_img_file=attr_img_file,
        )
        image_to_split = {}
        for line in _iter_data_lines(partition_file):
            tokens = line.split()
            if len(tokens) < 2:
                continue
            image_name, split_name = tokens[0], tokens[1]
            if split_name not in {"train", "val", "test"}:
                continue
            image_to_split[image_name] = split_name

        for image_name, split_name in image_to_split.items():
            if image_name not in image_to_label or image_name not in image_to_attrs:
                continue
            self.samples.append(
                (image_name, image_to_label[image_name], image_to_attrs[image_name])
            )
            self.sample_splits.append(split_name)

    def _load_coarse_annotations(self, category_img_file, attr_img_file):
        image_to_label = {}
        for line in _iter_data_lines(category_img_file):
            tokens = line.split()
            if len(tokens) < 2:
                continue
            image_name = tokens[0]
            raw_label = int(tokens[-1])
            image_to_label[image_name] = raw_label - 1  # benchmark labels are 1-based

        image_to_attrs = {}
        for line in _iter_data_lines(attr_img_file):
            tokens = line.split()
            if len(tokens) < 2:
                continue
            image_name, raw_values = tokens[0], tokens[1:]
            image_to_attrs[image_name] = torch.tensor(
                [float(v) for v in raw_values], dtype=torch.float32
            )

        return image_to_label, image_to_attrs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, class_id, raw_attributes = self.samples[idx]
        img_path = self.path / image_name

        img = Image.open(img_path).convert("RGB")
        W_orig, H_orig = img.size

        bbox = self.image_to_bbox.get(image_name, None)

        if self.transform is not None:
            img = self.transform(img)

            if bbox is not None and torch.is_tensor(img) and img.ndim == 3:
                _, H_new, W_new = img.shape
                scale_x = float(W_new) / float(W_orig)
                scale_y = float(H_new) / float(H_orig)

                bbox = bbox.clone()
                bbox[0] = bbox[0] * scale_x
                bbox[2] = bbox[2] * scale_x
                bbox[1] = bbox[1] * scale_y
                bbox[3] = bbox[3] * scale_y

        attributes = (self._select_attributes(raw_attributes) > 0).to(torch.float32)

        sample = {
            "image": img,
            "label": torch.tensor(class_id, dtype=torch.int64),
        }

        if bbox is not None:
            sample["bbox"] = bbox

        sample.update(
            {attr_key: attributes[i] for i, attr_key in enumerate(self.attribute_keys)}
        )

        return sample

class DeepFashion(DatasetWrap):
    """
    DeepFashion loader built from the official coarse annotation files.

    Args:
        path (str or pathlib.Path): Root directory of the DeepFashion dataset.
        transform (callable, optional): Transform applied to validation and test samples. Defaults to `vgg16_imagenet`.
        augmentation (callable, optional): If provided, this transform is used for train split 
        seed (int, optional): Random seed stored with the dataset wrapper.
        top_k_attributes (int, optional): Restrict returned attributes to the k most frequent attributes 
            If None, all attributes are kept (DeepFashion has 1000 attributes originally).
            Returned attribute values are binary indicators, where positive
            values from `list_attr_img.txt` become `1.0` and non-positive
            values become `0.0`.
        del_no_attributes_samples (bool, optional): When True, removes samples that have 0 positive attributes.
        min_samples_per_class (int, optional): Minimum number of samples a class must have in each of the train, validation, and test splits
            to be kept. Remaining classes are remapped to contiguous labels.
    """

    def __init__(self, **kwargs):
        self.path = kwargs.get("path")
        self.transform = kwargs.get("transform", vgg16_transform)
        self.augmentation = kwargs.get("augmentation", None)
        self.seed = kwargs.get("seed", 42)
        self.top_k_attributes = kwargs.get("top_k_attributes", None)
        self.del_no_attributes_samples = bool(
            kwargs.get("del_no_attributes_samples", False)
        )
        self.min_samples_per_class = int(kwargs.get("min_samples_per_class", 1))
        self.label_map = None
        assert self.min_samples_per_class >= 1

    def _group_indices_by_label(self, indices, source_ds):
        '''
        Builds label -> [indices belonging to that label]
        This is used for filtering out classes that don't have enough samples in each split.
        '''
        groups = defaultdict(list)
        for idx in indices:
            label = int(source_ds.samples[idx][1])
            groups[label].append(idx)
        return groups

    def _remap_dataset_labels(self, ds, label_map, new_id_to_class=None):
        '''
        Remaps the labels of a dataset according to label_map.
        '''
        remapped_samples = []
        for image_name, old_label, raw_attributes in ds.samples:
            new_label = label_map.get(old_label, old_label)
            remapped_samples.append((image_name, new_label, raw_attributes))
        ds.samples = remapped_samples

        if new_id_to_class is not None:
            ds.id_to_class = dict(new_id_to_class)
            return

        remapped_id_to_class = {}
        for old_label, class_name in ds.id_to_class.items():
            new_label = label_map.get(old_label, old_label)
            if new_label not in remapped_id_to_class:
                remapped_id_to_class[new_label] = class_name
        if len(remapped_id_to_class) > 0:
            ds.id_to_class = remapped_id_to_class

    def _filter_classes_by_min_samples(self, base_ds, train_indices, val_indices, test_indices):
        '''
        Filter out classes that don't have at least min_samples_per_class in each split.
        '''
        min_samples = self.min_samples_per_class
        self.balance_thresholds = {"min": {"train": min_samples, "val": min_samples, "test": min_samples}}

        train_groups = self._group_indices_by_label(train_indices, base_ds)
        val_groups = self._group_indices_by_label(val_indices, base_ds)
        test_groups = self._group_indices_by_label(test_indices, base_ds)

        train_ok = {k for k, v in train_groups.items() if len(v) >= min_samples}
        val_ok = {k for k, v in val_groups.items() if len(v) >= min_samples}
        test_ok = {k for k, v in test_groups.items() if len(v) >= min_samples}
        keep_classes = sorted(train_ok.intersection(val_ok).intersection(test_ok))

        keep_set = set(keep_classes)
        train_indices = [i for i in train_indices if int(base_ds.samples[i][1]) in keep_set]
        val_indices = [i for i in val_indices if int(base_ds.samples[i][1]) in keep_set]
        test_indices = [i for i in test_indices if int(base_ds.samples[i][1]) in keep_set]

        return train_indices, val_indices, test_indices, keep_classes

    def __load_data__(self, **kwargs):
        """
        Load DeepFashion and build train/val/test subsets.
        """
        self.__dataset__ = {}

        base_ds = CustomDS(
            path=self.path,
            transform=self.transform,
            seed=self.seed,
            top_k_attributes=self.top_k_attributes,
            del_no_attributes_samples=self.del_no_attributes_samples,
        )

        if self.augmentation is None:
            train_source_ds = base_ds
        else:
            train_source_ds = CustomDS(
                path=self.path,
                transform=self.augmentation,
                seed=self.seed,
                top_k_attributes=self.top_k_attributes,
                del_no_attributes_samples=self.del_no_attributes_samples,
            )

        train_indices = [
            i for i, split_tag in enumerate(base_ds.sample_splits) if split_tag == "train"
        ]
        val_indices = [
            i for i, split_tag in enumerate(base_ds.sample_splits) if split_tag == "val"
        ]
        test_indices = [
            i for i, split_tag in enumerate(base_ds.sample_splits) if split_tag == "test"
        ]

        # filter out classes that don't have enough samples 
        train_indices, val_indices, test_indices, active_class_ids_raw = self._filter_classes_by_min_samples(
            base_ds,
            train_indices,
            val_indices,
            test_indices,
        )
        label_map = None
        # make sure label map is contiguous (avoid things like [0,1,4,7])
        if label_map is None:
            sorted_active = sorted(active_class_ids_raw)
            if sorted_active != list(range(len(sorted_active))):
                label_map = {
                    old_label: new_label
                    for new_label, old_label in enumerate(sorted_active)
                }
            else:
                label_map = {k: k for k in sorted_active}

        # expose metadata for downstream use
        self.id_to_class = {
            label_map[old_label]: base_ds.id_to_class[old_label]
            for old_label in sorted(active_class_ids_raw)
        }
        self.label_map = dict(label_map)
        self.label_map_old_to_new = dict(self.label_map)
        self.label_map_new_to_old = {
            new_label: old_label for old_label, new_label in self.label_map.items()
        }
        self.label_map_original_to_new = dict(self.label_map_old_to_new)
        self.active_class_ids_raw = sorted(active_class_ids_raw)

        self.active_class_ids = sorted(self.id_to_class.keys())
        self.num_classes = len(self.id_to_class)
        self.attribute_names = base_ds.attribute_names
        self.attribute_keys = base_ds.attribute_keys

        # Apply remapping to underlying datasets so downstream code that directly
        self._remap_dataset_labels(base_ds, self.label_map_old_to_new, new_id_to_class=self.id_to_class)
        if train_source_ds is not base_ds:
            self._remap_dataset_labels(train_source_ds, self.label_map_old_to_new, new_id_to_class=self.id_to_class)

        self.__dataset__ = {
            "DeepFashion-train": Subset(train_source_ds, train_indices),
            "DeepFashion-val": Subset(base_ds, val_indices),
            "DeepFashion-test": Subset(base_ds, test_indices),
        }

    def get(self, ds_key, idx):
        if not self.__dataset__:
            raise RuntimeError("Data not loaded. Please run load_data() first.")
        return [self.__dataset__[ds_key][idx]]
