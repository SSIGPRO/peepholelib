# general python stuff
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.functional.transforms import vgg16_imagenet

# torch stuff
import torch
from torch.utils.data import Dataset, Subset

DEFAULT_DEEPFASHION_CLASS_MERGE_MAP = {
    "Anorak": "Coat",
    "Blazer": "Coat",
    "Blouse": "Button-Down",
    "Bomber": "Coat",
    "Button-Down": "Button-Down",
    "Cardigan": "Coat",
    "Flannel": "Button-Down",
    "Halter": "Halter",
    "Henley": "Long-sleeve shirt",
    "Hoodie": "Long-sleeve shirt",
    "Jacket": "Coat",
    "Jersey": "Short-sleeve shirt",
    "Parka": "Coat",
    "Peacoat": "Coat",
    "Poncho": "Poncho",
    "Sweater": "Long-sleeve shirt",
    "Tank": "Short-sleeve shirt",
    "Tee": "Short-sleeve shirt",
    "Top": "Short-sleeve shirt",
    "Turtleneck": "Long-sleeve shirt",
    "Capris": "Pants",
    "Chinos": "Pants",
    "Culottes": "Pants",
    "Cutoffs": "Shorts",
    "Gauchos": "Pants",
    "Jeans": "Pants",
    "Jeggings": "Pants",
    "Jodhpurs": "Pants",
    "Joggers": "Pants",
    "Leggings": "Pants",
    "Sarong": "Sarong",
    "Shorts": "Shorts",
    "Skirt": "Skirt",
    "Sweatpants": "Pants",
    "Sweatshorts": "Shorts",
    "Trunks": "Shorts",
    "Caftan": "Dress",
    "Cape": "Cape",
    "Coat": "Coat",
    "Coverup": "Coverup",
    "Dress": "Dress",
    "Jumpsuit": "Jumpsuit",
    "Kaftan": "Dress",
    "Kimono": "Dress",
    "Nightdress": "Dress",
    "Onesie": "Onesie",
    "Robe": "Robe",
    "Romper": "Romper",
    "Shirtdress": "Dress",
    "Sundress": "Dress",
}


def _normalize_class_name(class_name):
    normalized = re.sub(r"[^0-9a-zA-Z]+", " ", str(class_name).strip().lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


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


def _safe_attr_key(idx, name, used_keys):
    base_name = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    if not base_name:
        base_name = "attribute"
    key = f"attr_{idx:04d}_{base_name}"
    suffix = 1
    while key in used_keys:
        suffix += 1
        key = f"attr_{idx:04d}_{base_name}_{suffix}"
    used_keys.add(key)
    return key


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
        self.attribute_encoding = kwargs.get("attribute_encoding", "binary")

        if self.attribute_encoding not in {"binary", "raw"}:
            raise ValueError("attribute_encoding must be one of {'binary', 'raw'}.")

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

        self.n_attributes = len(self.attribute_names)

        used_keys = set()
        self.attribute_keys = [
            _safe_attr_key(i, name, used_keys)
            for i, name in enumerate(self.attribute_names)
        ]

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

    def _normalize_split_tag(self, split_name):
        split_name = split_name.strip().lower()
        if split_name in {"train", "training"}:
            return "train"
        if split_name in {"val", "valid", "validation"}:
            return "val"
        if split_name in {"test", "testing"}:
            return "test"
        return None

    def _parse_raw_attributes(self, values):
        raw_attrs = torch.tensor([float(v) for v in values], dtype=torch.float32)
        if raw_attrs.numel() < self.n_attributes:
            pad = torch.zeros(self.n_attributes - raw_attrs.numel(), dtype=torch.float32)
            raw_attrs = torch.cat([raw_attrs, pad], dim=0)
        elif raw_attrs.numel() > self.n_attributes:
            raw_attrs = raw_attrs[: self.n_attributes]
        return raw_attrs

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
            image_name, raw_split = tokens[0], tokens[1]
            split_name = self._normalize_split_tag(raw_split)
            if split_name is None:
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
            image_to_attrs[image_name] = self._parse_raw_attributes(raw_values)

        return image_to_label, image_to_attrs

    def _encode_attributes(self, raw_attributes):
        if self.attribute_encoding == "binary":
            return (raw_attributes > 0).to(torch.float32)
        return raw_attributes.to(torch.float32)

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

        attributes = self._encode_attributes(raw_attributes)

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
    DeepFashion has an unbalanced class sample count :/

    Optional:
    - merge_classes (bool): if True, merge classes together before balancing.
    - label_map (dict[int, int] | None): custom original->merged
    - balance_dataset (bool): if True, build balanced split indices.
    - min_samples_per_class (int): classes with less than min_samples_per_class are removed. 
    """

    def __init__(self, **kwargs):
        self.path = kwargs.get("path")
        self.transform = kwargs.get("transform", vgg16_imagenet)
        self.augmentation = kwargs.get("augmentation", None)
        self.seed = kwargs.get("seed", 42)
        self.attribute_encoding = kwargs.get("attribute_encoding", "binary")
        self.merge_classes = bool(kwargs.get("merge_classes", False))
        self.balance_dataset = kwargs.get("balance_dataset", False)
        self.min_samples_per_class = int(kwargs.get("min_samples_per_class", 1))
        self.merge_label_map = kwargs.get("label_map", None)
        self.label_map = None
        assert self.min_samples_per_class >= 1

    def _group_indices_by_label(self, indices, source_ds):
        groups = defaultdict(list)
        for idx in indices:
            label = int(source_ds.samples[idx][1])
            groups[label].append(idx)
        return groups

    def _merge_classes(self, base_ds, train_source_ds):
        '''
        This is the best solution I could find to fix the unblance in classes.
        Without it balance_classes() would either create too many duplicates or eliminate too many samples
        
        Merge classes together according to self.merge_label_map or DEFAULT_DEEPFASHION_CLASS_MERGE_MAP.
        '''
        if self.merge_label_map is not None:
            label_map_original_to_merged = {
                int(old_label): int(merged_label)
                for old_label, merged_label in dict(self.merge_label_map).items()
            }
            merged_id_to_class = {}
            for old_label in sorted(base_ds.id_to_class):
                merged_label = int(label_map_original_to_merged.get(old_label, old_label))
                if merged_label not in merged_id_to_class:
                    merged_id_to_class[merged_label] = base_ds.id_to_class[old_label]
        else:
            normalized_map = {
                _normalize_class_name(src): re.sub(r"\s+", " ", str(dst).replace("-", " ")).strip()
                for src, dst in DEFAULT_DEEPFASHION_CLASS_MERGE_MAP.items()
            }
            label_map_original_to_merged = {}
            merged_name_to_id = {}
            merged_id_to_class = {}
            for old_label in sorted(base_ds.id_to_class):
                source_name = base_ds.id_to_class[old_label]
                source_key = _normalize_class_name(source_name)
                target_name = normalized_map.get(source_key, source_name)
                if target_name == "":
                    target_name = source_name
                if target_name not in merged_name_to_id:
                    new_id = len(merged_name_to_id)
                    merged_name_to_id[target_name] = new_id
                    merged_id_to_class[new_id] = target_name
                label_map_original_to_merged[old_label] = merged_name_to_id[target_name]

        remapped_base_samples = []
        for image_name, old_label, raw_attributes in base_ds.samples:
            new_label = label_map_original_to_merged.get(old_label, old_label)
            remapped_base_samples.append((image_name, new_label, raw_attributes))
        base_ds.samples = remapped_base_samples
        base_ds.id_to_class = dict(merged_id_to_class)

        if train_source_ds is not base_ds:
            remapped_train_samples = []
            for image_name, old_label, raw_attributes in train_source_ds.samples:
                new_label = label_map_original_to_merged.get(old_label, old_label)
                remapped_train_samples.append((image_name, new_label, raw_attributes))
            train_source_ds.samples = remapped_train_samples
            train_source_ds.id_to_class = dict(merged_id_to_class)

        return label_map_original_to_merged

    def _remap_dataset_labels(self, ds, label_map, new_id_to_class=None):
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

    def _rebalance_indices(self, split_specs, keep_classes):
        '''
        Rebalance the given split indices to have equal class sample counts over the given keep_classes.
        1) compute class-count median over kept classes
        2) use the class count closest to that median as target
        3) classes above target are downsampled, below target are oversampled
        '''
        split_seed_offsets = {"train": 0, "val": 1, "test": 2}
        balanced_indices = {}
        balance_targets = {}

        for split_name, indices, source_ds in split_specs:
            groups = self._group_indices_by_label(indices, source_ds)
            counts = {cls_id: len(groups[cls_id]) for cls_id in keep_classes}
            effective_counts = dict(counts)

            sorted_counts = sorted(effective_counts.values())
            if len(sorted_counts) == 1:
                median_count = float(sorted_counts[0])
            else:
                n_counts = len(sorted_counts)
                if n_counts % 2 == 1:
                    median_count = float(sorted_counts[n_counts // 2])
                else:
                    hi = n_counts // 2
                    lo = hi - 1
                    median_count = 0.5 * (sorted_counts[lo] + sorted_counts[hi])

            target_class = min(keep_classes,
                key=lambda cls_id: (abs(effective_counts[cls_id] - median_count),
                    effective_counts[cls_id]),
            )
            target = effective_counts[target_class]
            generator = torch.Generator().manual_seed(
                self.seed + split_seed_offsets.get(split_name, 3)
            )

            balanced = []
            for cls_id in sorted(keep_classes):
                cls_indices = groups[cls_id]
                cls_size = len(cls_indices)
                if cls_size > target:
                    perm = torch.randperm(cls_size, generator=generator)[:target]
                    balanced.extend([cls_indices[i] for i in perm.tolist()])
                else:
                    perm = torch.randperm(cls_size, generator=generator)
                    balanced.extend([cls_indices[i] for i in perm.tolist()])
                if cls_size < target:
                    picks = torch.randint(
                        low=0,
                        high=cls_size,
                        size=(target - cls_size,),
                        generator=generator,
                    )
                    balanced.extend([cls_indices[i] for i in picks.tolist()])

            order = torch.randperm(len(balanced), generator=generator).tolist()
            balanced_indices[split_name] = [balanced[i] for i in order]
            balance_targets[split_name] = target

        return balanced_indices, balance_targets

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

    def balance_classes(self, base_ds, train_source_ds, train_indices, val_indices, test_indices, keep_classes):
        """
        Balance each split to equal class sample counts over already kept classes.
        """
        if len(keep_classes) == 0:
            return (train_indices, val_indices, test_indices, None, keep_classes, None)

        split_specs = [
            ("train", train_indices, train_source_ds),
            ("val", val_indices, base_ds),
            ("test", test_indices, base_ds),
        ]
        balanced_indices, balance_targets = self._rebalance_indices(split_specs, keep_classes)
        train_indices = balanced_indices["train"]
        val_indices = balanced_indices["val"]
        test_indices = balanced_indices["test"]

        label_map = {old: new for new, old in enumerate(keep_classes)}
        active_class_ids = keep_classes

        return (train_indices, val_indices, test_indices, label_map, active_class_ids, balance_targets)

    def __load_data__(self, **kwargs):
        """
        Load DeepFashion and build train/val/test subsets.
        """
        self.__dataset__ = {}

        base_ds = CustomDS(
            path=self.path,
            transform=self.transform,
            seed=self.seed,
            attribute_encoding=self.attribute_encoding,
        )

        if self.augmentation is None:
            train_source_ds = base_ds
        else:
            train_source_ds = CustomDS(
                path=self.path,
                transform=self.augmentation,
                seed=self.seed,
                attribute_encoding=self.attribute_encoding,
            )

        label_map_original_to_merged = None
        if self.merge_classes:
            label_map_original_to_merged = self._merge_classes(base_ds, train_source_ds)

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
        self.balance_targets = None
        
        # balance classes' sample count
        if self.balance_dataset:
            (
                train_indices,
                val_indices,
                test_indices,
                label_map,
                active_class_ids_raw,
                self.balance_targets,
            ) = self.balance_classes(
                base_ds,
                train_source_ds,
                train_indices,
                val_indices,
                test_indices,
                active_class_ids_raw,
            )
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
        self.label_map_original_to_new = None
        if label_map_original_to_merged is not None:
            self.label_map_original_to_new = {
                old_label: self.label_map[merged_label]
                for old_label, merged_label in label_map_original_to_merged.items()
                if merged_label in self.label_map
            }
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
