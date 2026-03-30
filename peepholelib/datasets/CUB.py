# general python stuff
from collections import defaultdict
from PIL import Image
from pathlib import Path

# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap

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
        with open(classes_file, "r") as f:
            for line in f:
                class_id, class_name = line.strip().split()
                self.class_id_to_name[int(class_id) - 1] = class_name

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
                attr_id = int(attr_id_str)
                is_present = bool(int(is_present_str))
                certainty = int(certainty_str)

                attr_info = {
                    "attribute_id": attr_id,
                    "attribute_name": self.attr_id_to_name.get(attr_id),
                    "is_present": is_present,
                    "certainty": certainty,
                }
                self.id_to_attributes[img_id].append(attr_info)

        self.id_to_attributes_categorical = {}

        for sample_id, sample_attributes in self.id_to_attributes.items():
            self.id_to_attributes_categorical[sample_id] = {}

            for attribute in attributes_list:
                encoding = []
                for sa in sample_attributes:
                    if attribute in sa['attribute_name']:
                        encoding.append(sa['is_present'])
                
                self.id_to_attributes_categorical[sample_id][attribute] = onehot_to_index(encoding)

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
        attributes_categorical = self.id_to_attributes_categorical.get(img_id, [])

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
        self.transform = kwargs.get('std_transform', None)
        self.augmentation = kwargs.get('aug_transform', None)
        self.train_ratio = kwargs.get('train_ratio', 0.8)
        self.seed = kwargs.get('seed', 42)
        self.reference_ds = kwargs.get('reference_ds', None)

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

    def __load_data__(self, **kwargs):
        """
        Load and prepare CUB data.
        """
        self.__dataset__ = {}

        base_ds = CUBCustom(
            path=self.path,
            transform=self.transform,
            reference_ds=self.reference_ds,
            seed=self.seed
        )

        train_indices_all = [i for i, flag in enumerate(base_ds.is_train) if flag == 1]
        test_indices = [i for i, flag in enumerate(base_ds.is_train) if flag == 0]

        train_indices, val_indices = random_split(
                train_indices_all,
                [self.train_ratio, 1.0 - self.train_ratio],
                generator=torch.Generator().manual_seed(self.seed)
                )

        train_source_ds = base_ds
        if self.augmentation is not None:
            train_source_ds = CUBCustom(
                path=self.path,
                transform=self.augmentation,
                reference_ds=self.reference_ds,
                seed=self.seed
            )

        self.__dataset__['CUB-train'] = Subset(train_source_ds, train_indices)
        self.__dataset__['CUB-val'] = Subset(base_ds, val_indices)
        self.__dataset__['CUB-test'] = Subset(base_ds, test_indices)
