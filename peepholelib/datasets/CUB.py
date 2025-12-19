# general python stuff
import os
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from types import NoneType
from math import floor, ceil 

# Our stuff
# from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.models.prediction_fns import multilabel_classification

# torch stuff
import torch
from torch.utils.data import Dataset #, DataLoader
# from torch.utils.data import random_split
# from tensordict import PersistentTensorDict
# from tensordict import MemoryMappedTensor as MMT

def onehot_to_index(bits):
    for i, b in enumerate(bits):
        if b == 1:
            return torch.tensor([i])
    return torch.tensor([0])

class CustomDS(Dataset):

    def __init__(self, path, train=True, transform=None):
        """
        path: path to CUB_200_2011 (folder that contains images/, attributes/, parts/, *.txt)
        train: True -> train split, False -> test split (uses train_test_split.txt)
        """
        Dataset.__init__(self) 
        self.path = Path(path)
        self.transform = transform

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
                self.id_to_label[int(img_id)] = torch.tensor([int(class_id) - 1])

        # ---- 4) train / test split ----
        # train_test_split.txt: <image_id> <is_training_image>
        self.train_ids = []
        self.test_ids = []
        with open(split_file, "r") as f:
            for line in f:
                img_id, is_train = line.strip().split()
                img_id = int(img_id)
                is_train = int(is_train)
                if is_train == 1:
                    self.train_ids.append(img_id)
                else:
                    self.test_ids.append(img_id)

        self.img_ids = self.train_ids if train else self.test_ids

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

                    if attribute in sa['attribute_name']: encoding.append(sa['is_present'])
                
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
        self.path = kwargs.get('path')
        self.transform = kwargs.get('transform')
        return

    def __load_data__(self, **kwargs):
        """
        Load and prepare CUB data.
        """
        self.__dataset__ = {}

        # Load train split
        train_dataset = CustomDS(
            path=self.path,
            train=True,
            transform=self.transform
        )

        # Load test split
        test_dataset = CustomDS(
            path=self.path,
            train=False,
            transform=self.transform
        )

        self.__dataset__ = {
            "train": train_dataset,
            "test": test_dataset
        }

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
    
# class CUB(ParsedDataset):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         print(self.path)
#         return
    
    
#     def create_ds(self, **kwargs):
#         path = Path(kwargs['path'])
#         cub_wrap = kwargs['cub_wrap']
#         bs = kwargs.get('batch_size', 2**11)
#         verbose = kwargs.get('verbose', False)

#         cub_wrap.__load_data__()
        
#         path.mkdir(parents=True, exist_ok=True)
#         self._dss = {}
        
#         for ds_key in cub_wrap.__dataset__:
#             file_path = self.path/('dss.'+ds_key)
#             n_samples = len(cub_wrap.__dataset__[ds_key])
            
#             # check if PTD exists 
#             if file_path.exists():
#                 if verbose: print(f'File {file_path} exists. Loading from disk.')
#                 self._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

#             else:
#                 if verbose: print(f'Creating {ds_key} dataset with n_samples: ', n_samples)
#                 self._dss[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
                
#                 # get sample to get shapes
#                 sample = cub_wrap.__dataset__[ds_key][0]
#                 for key in sample.keys():
#                     if verbose: print(f'allocating {key} with shape {sample[key].shape}')
#                     self._dss[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,)+sample[key].shape), dtype=torch.float32)
        
#                 # create Dataloader of input dataset
#                 ds_in = DataLoader(
#                     dataset = cub_wrap.__dataset__[ds_key],
#                     batch_size = bs
#                 )
                                                                                                                                        
#                 ds_t = DataLoader(
#                     self._dss[ds_key],
#                     collate_fn = lambda x:x, 
#                     batch_size = bs
#                 )
                                                                                                                                        
#                 for data_in, data_t in tqdm(zip(ds_in, ds_t), disable=not verbose, total=ceil(n_samples/bs)):
#                     for key in data_in.keys():
#                         data_t[key] = data_in[key]
            
#             # close the PTD
#             self._dss[ds_key].close()
#         return

#     # overwrite the parse_ds() from peepholelib.datasets.parsedDataset.ParsedDataset
#     def parse_ds(self, **kwargs):
#         self.check_uncontexted()

#         model = kwargs['model']
#         loaders = kwargs.get('loaders', None)
#         bs = kwargs.get('batch_size', 2**11)
#         verbose = kwargs.get('verbose', False)
#         pred_fn = kwargs.get('pred_fn', multilabel_classification)
        
#         if loaders == None:
#             loaders = self._dss.keys()

#         for ds_key in loaders:
#             file_path = self.path/('dss.'+ds_key)
#             n_samples = len(self._dss[ds_key])
            
#             # check if PTD exists 
#             if file_path.exists():
#                 if verbose: print(f'File {file_path} exists. Loading from disk.')
#                 self._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
#             else:
#                 if verbose: raise RuntimeError(f'Not {ds_key} dataset run create_ds first')

#             if ('output' in self._dss[ds_key]) and ('pred' in self._dss[ds_key]) and ('result' in self._dss[ds_key]):
#                 if verbose: print(f'keys: output, pred and result already parsed for {ds_key}. check the next')
#                 continue
            
#             # dataset sample for dry run
#             sample = self._dss[ds_key][0:1]['image'].to(model.device)
#             with torch.no_grad():
#                 _out = model(sample)
                     
#             os = _out.shape[1:]

#             # need to fix the batch size - workaround  
#             self._dss[ds_key].batch_size = torch.Size((n_samples,))
#             # allocate disk space
#             self._dss[ds_key]['output'] = MMT.empty(shape=torch.Size((n_samples,)+os), dtype=torch.float32)
#             self._dss[ds_key]['pred'] = MMT.empty(shape=torch.Size((n_samples,)))
#             self._dss[ds_key]['result'] = MMT.empty(shape=torch.Size((n_samples,)))
#             print(self._dss[ds_key]['pred'].shape, self._dss[ds_key]['label'].shape)

#             dl = DataLoader(
#                 self._dss[ds_key],
#                 collate_fn = lambda x:x, 
#                 batch_size = bs
#             )

#             for data in tqdm(dl, disable=not verbose, total=ceil(n_samples/bs)):
#                 #compute predictions which is the out of decoder

#                 with torch.no_grad():
#                     y_predicted = model(data['image'].to(model.device))
            
#                     predicted_labels = pred_fn(y_predicted).detach().cpu()
#                     data['output'] = y_predicted
#                     data['pred'] = predicted_labels
#                     data['result'] = predicted_labels == data['label'].squeeze(dim=1)         
        
#         return
