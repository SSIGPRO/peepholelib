# general python stuff
from collections import defaultdict
from PIL import Image
from pathlib import Path

# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.functional.transforms import vgg16_transform 

# torch stuff
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import ToTensor

def onehot_to_index(bits):
    for i, b in enumerate(bits):
        if b == 1:
            return torch.tensor([i])
    return torch.tensor([0])

class CUBCustom(Dataset):
    def __init__(self, **kwargs):
        """
        path: path to CUB_200_2011 (folder that contains images/, attributes/, parts/, *.txt)
        """
        Dataset.__init__(self) 
        self.path = Path(kwargs['path'])
        self.transform = kwargs['transform']
        self.seed = kwargs['seed']
        self.collapse_attributes = bool(kwargs.get('collapse_attributes', True))
        self.collapse_attribute_kinds = kwargs.get(
            'collapse_attribute_kinds',
            ('color', 'shape', 'pattern'),
        )

        if isinstance(self.collapse_attribute_kinds, str):
            if self.collapse_attribute_kinds.lower() == 'all':
                self.collapse_attribute_kinds = None
            else:
                self.collapse_attribute_kinds = {self.collapse_attribute_kinds.lower()}
        elif self.collapse_attribute_kinds is not None:
            self.collapse_attribute_kinds = {
                str(kind).lower() for kind in self.collapse_attribute_kinds
            }

        # ---- 1) basic files ----
        images_file = self.path / "images.txt"
        labels_file = self.path / "image_class_labels.txt"
        split_file = self.path / "train_test_split.txt"
        classes_file = self.path / "classes.txt"
        bbox_file = self.path / "bounding_boxes.txt"

        # ---- 2) load image paths ----
        # images.txt: <image_id> <relative_path>
        self.id_to_relpath = {}
        with open(images_file, "r") as f:
            for line in f:
                img_id, rel_path = line.strip().split()
                self.id_to_relpath[int(img_id)] = rel_path

        # ---- 3) class labels ----
        # image_class_labels.txt: <image_id> <class_id>
        self.id_to_label = {}
        with open(labels_file, "r") as f:
            for line in f:
                img_id, class_id = line.strip().split()
                # make labels 0-based
                self.id_to_label[int(img_id)] = torch.tensor(int(class_id) - 1, dtype=torch.int64)

        # ---- 4) img ids ----
        self.img_ids = []
        self.is_train = []
        with open(split_file, "r") as f:
            for line in f:
                img_id, is_train = line.strip().split()
                img_id = int(img_id)
                is_train = int(is_train)
                
                self.img_ids.append(img_id)
                self.is_train.append(is_train)

        # ---- 5) class names ----
        # classes.txt: <class_id> <class_name>
        self.class_id_to_name = {}
        self.id_to_class = {}
        with open(classes_file, "r") as f:
            for line in f:
                class_id, class_name = line.strip().split()
                class_id = int(class_id)
                self.class_id_to_name[class_id - 1] = class_name
                self.id_to_class[class_id - 1] = class_name

        # ---- 6) bounding boxes ----
        # bounding_boxes.txt: <image_id> <x> <y> <width> <height>
        self.id_to_bbox = {}
        with open(bbox_file, "r") as f:
            for line in f:
                img_id, x, y, w, h = line.strip().split()
                self.id_to_bbox[int(img_id)] = torch.tensor([float(x), float(y), float(w), float(h)])

        # ---- 7) parts info ----
        # parts/parts.txt: <part_id> <part_name>

        parts_dir = self.path / "parts"
        parts_file = parts_dir / "parts.txt"
        part_locs_file = parts_dir / "part_locs.txt"

        self.part_id_to_name = {}
        with open(parts_file, "r") as f:
            for line in f:
                tokens = line.strip().split()
                part_id = int(tokens[0])
                part_name = " ".join(tokens[1:])  # handles multi-word names like "left eye"
                self.part_id_to_name[part_id] = part_name

        # parts per image: dict[image_id] -> list of dicts
        self.id_to_parts = defaultdict(list)
        # part_locs.txt: <image_id> <part_id> <x> <y> <visible>
        with open(part_locs_file, "r") as f:
            for line in f:
                img_id, part_id, x, y, visible = line.strip().split()
                img_id = int(img_id)
                part_id = int(part_id)
                part_info = {
                    "part_id": part_id,
                    "part_name": self.part_id_to_name.get(part_id),
                    "x": float(x),
                    "y": float(y),
                    "visible": bool(visible),
                }
                self.id_to_parts[img_id].append(part_info)
        
        self.id_to_parts_categorical = {}
        for sample_id, sample_parts in self.id_to_parts.items():
            self.id_to_parts_categorical[sample_id] = {}
            for part in sample_parts:
                
                self.id_to_parts_categorical[sample_id][part['part_name']] = torch.tensor([part['x'], part['y'], part['visible']])

        # ---- 8) attributes ----
        # attributes/attributes.txt: <attribute_id> <attribute_name>
        attr_dir = self.path / "attributes"
        attr_file = attr_dir / "attributes.txt"
        image_attr_file = attr_dir / "image_attribute_labels.txt"

        self.attr_id_to_name = {}
        with open(attr_file, "r") as f:
            for line in f:
                attr_id, attr_name = line.strip().split(None, 1)
                self.attr_id_to_name[int(attr_id)] = attr_name.strip()

        # attributes per image: dict[image_id] -> list of dicts
        self.id_to_attributes = defaultdict(list)

        attributes_list = [
                'has_bill_shape', 
                'has_wing_color',
                'has_upperparts_color',
                'has_underparts_color',
                'has_breast_pattern',
                'has_back_color',
                'has_tail_shape',
                'has_upper_tail_color',
                'has_head_pattern',
                'has_breast_color',
                'has_throat_color',
                'has_eye_color',
                'has_bill_length',
                'has_forehead_color',
                'has_under_tail_color',
                'has_nape_color',
                'has_belly_color',
                'has_wing_shape',
                'has_size',
                'has_shape',
                'has_back_pattern',
                'has_tail_pattern',
                'has_belly_pattern',
                'has_primary_color',
                'has_leg_color',
                'has_bill_color',
                'has_crown_color',
                'has_wing_pattern'
                ]

        with open(image_attr_file, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 5:
                    continue  # skip empty / bad lines

                # take only the first 5 columns, ignore anything extra
                img_id_str, attr_id_str, is_present_str, certainty_str, _ = tokens[:5]

                img_id = int(img_id_str)
                raw_attr_id = int(attr_id_str)
                collapsed_attr_id = raw_attr_id_to_collapsed_id[raw_attr_id]
                is_present = bool(int(is_present_str))
                certainty = int(certainty_str)

                stats = per_image_collapsed_stats[img_id][collapsed_attr_id]
                stats["is_present"] = int(stats["is_present"] or int(is_present))
                stats["certainty_sum"] += float(certainty)
                stats["count"] += 1

        # one binary entry per collapsed concept
        self.attribute_names = [
            self.attr_id_to_name[attr_id]
            for attr_id in sorted(self.attr_id_to_name.keys())
        ]
        self.id_to_attributes = defaultdict(list)
        self.id_to_attributes_categorical = {}

        for sample_id, sample_attributes in self.id_to_attributes.items():
            self.id_to_attributes_categorical[sample_id] = {}

            for attribute in attributes_list:
                encoding = []
                for sa in sample_attributes:
                    if attribute in sa['attribute_name']:
                        encoding.append(sa['is_present'])
                
                self.id_to_attributes_categorical[sample_id][attribute] = onehot_to_index(encoding)

    def _collapse_attribute_name(self, attribute_name):
        if not self.collapse_attributes:
            return attribute_name

        attr_name = str(attribute_name).strip()
        if "::" not in attr_name:
            return attr_name

        left, right = attr_name.split("::", 1)
        left_tokens = [t for t in left.strip().split("_") if t]
        if len(left_tokens) == 0:
            return attr_name

        if left_tokens[0].lower() == "has":
            left_tokens = left_tokens[1:]

        if len(left_tokens) == 0:
            return attr_name

        concept_kind = left_tokens[-1].lower()

        if self.collapse_attribute_kinds is not None:
            if concept_kind not in self.collapse_attribute_kinds:
                return attr_name

        return f"{concept_kind}::{right.strip()}"

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        rel_path = self.id_to_relpath[img_id]
        img_dir = self.path / "images"
        img_path = img_dir / rel_path
        img = Image.open(img_path).convert("RGB")

        label = self.id_to_label[img_id]
        bbox = self.id_to_bbox.get(img_id, None)
        parts_categorical = self.id_to_parts_categorical.get(img_id, [])
        attributes_categorical = self.id_to_attributes_categorical.get(img_id, {})

        if self.transform is not None:

            x, y, w, h = bbox.tolist()

            W_orig, H_orig = img.size

            img = self.transform(img)

            _, W_new, H_new = img.shape

            scale_x = W_new / W_orig
            scale_y = H_new / H_orig

            bbox = torch.tensor([x * scale_x, y * scale_y, w  * scale_x, h * scale_y])

            scaled_parts = {}
            for name, t in parts_categorical.items():
                x, y, vis = t.tolist()
                x *= scale_x
                y *= scale_y
                scaled_parts[name] = torch.tensor([x, y, vis])

            parts_categorical = scaled_parts
            
        sample = {
            "image": img,
            "label": label,
            "bbox": bbox,
            **attributes_categorical,
            **parts_categorical
        }
        return sample
    
class CUB(DatasetWrap):

    def __init__(self, **kwargs):
        '''
        CUB loader (train & val & test). Train/val are created from the official CUB training split using `train_ratio`, while test uses the official CUB test split.

        Args:
            path (str): Path to the `CUB_200_2011` folder containing dataset files (e.g. `images/`, `attributes/`, `parts/`, and split/label `.txt` metadata files).
            transform (callable, optional): Transform applied to validation/test images. Defaults to `vgg16`.
            augmentation (callable, optional): If provided, applied only to the training split.
            train_ratio (float, optional): Fraction of official training samples used for train (remainder goes to val). Must be in (0, 1).
            seed (int, optional): Random seed used for deterministic train/val splitting.
            reference_ds (str, optional): Reserved optional argument for compatibility with other dataset loaders.
        Returns:
            - a thumbs up
        '''
        self.path = kwargs.get('path')
        self.transform = kwargs.get('transform', vgg16_transform)
        self.augmentation = kwargs.get('augmentation', None)
        self.merge_classes = bool(kwargs.get('merge_classes', False))
        self.train_ratio = kwargs.get('train_ratio', 0.8)
        self.seed = kwargs.get('seed', 42)
        self.reference_ds = kwargs.get('reference_ds', None)
        self.collapse_attributes = bool(kwargs.get('collapse_attributes', True))
        self.collapse_attribute_kinds = kwargs.get(
            'collapse_attribute_kinds',
            ('color', 'shape', 'pattern'),
        )

        assert 0.0 < self.train_ratio < 1.0

        # append ToTensor to the transform
        if self.transform != None:
            self.transform.transforms.append(ToTensor())
        else:
            self.transform = ToTensor()
                                                                          
        # if augmentation == None, transform will be used for all loaders
        if self.augmentation != None:
            self.augmentation.transforms.append(ToTensor())

        return

    def _class_group_name(self, class_name):
        name = str(class_name).strip()
        if '.' in name:
            name = name.split('.', 1)[1]
        name = name.strip()
        if not name:
            return str(class_name).strip()
        return name.split('_')[-1]

    def _build_group_merge_map(self, ds):
        label_map = {}
        group_name_to_new_id = {}
        merged_id_to_class = {}

        for old_label in sorted(ds.id_to_class.keys()):
            source_name = ds.id_to_class[old_label]
            target_name = self._class_group_name(source_name)

            if target_name not in group_name_to_new_id:
                new_id = len(group_name_to_new_id)
                group_name_to_new_id[target_name] = new_id
                merged_id_to_class[new_id] = target_name

            label_map[old_label] = group_name_to_new_id[target_name]

        return label_map, merged_id_to_class

    def _remap_dataset_labels(self, ds, label_map, new_id_to_class):
        remapped = {}
        for img_id, label in ds.id_to_label.items():
            old_label = int(label.item())
            new_label = int(label_map.get(old_label, old_label))
            remapped[img_id] = torch.tensor(new_label, dtype=torch.int64)

        ds.id_to_label = remapped
        ds.id_to_class = dict(new_id_to_class)
        ds.class_id_to_name = dict(new_id_to_class)

    def __load_data__(self, **kwargs):
        """
        Load and prepare CUB data.
        """
        generator = torch.Generator().manual_seed(self.seed)
        verbose = kwargs.get('verbose', False)
        self.__dataset__ = {}

        base_ds = CustomDS(
            path=self.path,
            transform=self.transform,
            reference_ds=self.reference_ds,
            seed=self.seed,
            collapse_attributes=self.collapse_attributes,
            collapse_attribute_kinds=self.collapse_attribute_kinds,
        )

        if verbose:
            n_attributes = len(base_ds.attribute_names)
            print(f'CUB attributes after collapsing: {n_attributes}')
            for i, attribute_name in enumerate(base_ds.attribute_names, start=1):
                print(f'{i} {attribute_name}')

        train_indices_all = [i for i, flag in enumerate(base_ds.is_train) if flag == 1]
        test_indices = [i for i, flag in enumerate(base_ds.is_train) if flag == 0]

        train_indices, val_indices = random_split(
                train_indices_all,
                [self.train_ratio, 1.0 - self.train_ratio],
                generator=torch.Generator().manual_seed(self.seed)
                )

        train_source_ds = base_ds
        if self.augmentation is not None:
            train_source_ds = CustomDS(
                path=self.path,
                transform=self.augmentation,
                reference_ds=self.reference_ds,
                seed=self.seed,
                collapse_attributes=self.collapse_attributes,
                collapse_attribute_kinds=self.collapse_attribute_kinds,
            )

        if self.merge_classes:
            label_map, merged_id_to_class = self._build_group_merge_map(base_ds)
            self._remap_dataset_labels(base_ds, label_map, merged_id_to_class)
            if train_source_ds is not base_ds:
                self._remap_dataset_labels(train_source_ds, label_map, merged_id_to_class)

            self.id_to_class = dict(merged_id_to_class)
            self.label_map = dict(label_map)
        else:
            self.id_to_class = dict(base_ds.id_to_class)
            self.label_map = {old_label: old_label for old_label in sorted(base_ds.id_to_class)}

        self.num_classes = len(self.id_to_class)

        self.__dataset__['CUB-train'] = Subset(train_source_ds, train_indices)
        self.__dataset__['CUB-val'] = Subset(base_ds, val_indices)
        self.__dataset__['CUB-test'] = Subset(base_ds, test_indices)

    def get(self, ds_key, idx):
        '''
        Get item from the dataset.
        
        Args:
        - idx (int): Index of the item to get.
        - ds_key (str): Key of the dataset to get the item from ('train', 'val', 'test').
        
        Returns:
        - a tuple of (image, label)
        '''
        if not self.__dataset__:
            raise RuntimeError('Data not loaded. Please run load_data() first.')
        
        return [self.__dataset__[ds_key][idx]]
