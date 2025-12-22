# general python stuff
# from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from pathlib import Path
# from types import NoneType
# import json
# import re

# Our stuff
# from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.datasets.datasetWrap import DatasetWrap
# from peepholelib.datasets.functional.transforms import vgg16_cifar100
# from peepholelib.models.prediction_fns import multilabel_classification

# torch stuff
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT

class CustomDS(Dataset):
    def __init__(self, **kwargs):
        """
        path: path to AwA2 folder containing JPEGImages/, classes.txt, predicate-matrix-binary.txt
        split: "train" or "test"
        train_ratio: fraction of samples used for training
        seed: for deterministic split
        """

        self.path = Path(kwargs['path'])
        self.reference_ds = kwargs['reference_ds']
        self.transform = kwargs['transform']        
        self.seed = kwargs['seed']

        # ---- Load class names ----
        classes_file = self.path / "classes.txt"
        self.id_to_class = {}
        with open(classes_file) as f:
            for line in f:
                cid, cname = line.strip().split('\t')
                self.id_to_class[int(cid)] = cname

        # ---- Load attribute matrix ----
        attr_file = self.path / "predicate-matrix-binary.txt"
        attr_list = []
        with open(attr_file, "r") as f:
            for line in f:
                attr_list.append([float(x) for x in line.strip().split()])
        self.attributes = torch.tensor(attr_list, dtype=torch.float32)  # [50, 85]

        # ---- Load attribute names ----
        pred_file = self.path / "predicates.txt"
        self.attribute_names = []
        with open(pred_file, "r") as f:
            for line in f:
                _, name = line.strip().split('\t')
                self.attribute_names.append(name)

        assert len(self.attribute_names) == self.attributes.shape[1]

        # ---- Build full list of samples (image_path, class_id) ----
        self.samples = []
        img_path = self.path / "JPEGImages"

        for cid, cname in self.id_to_class.items():
            class_dir = img_path / cname
            if not class_dir.is_dir():
                print(f"WARNING: folder missing → {class_dir}")
                continue

            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_file, cid - 1))
        
        if self.reference_ds is not None:

            # ---- Reference ds classes names ----

            if self.reference_ds == 'ImageNet':

                self.mapping_AwA_ImageNet = {
                                                0: [351, 352, 353],
                                                1: [294, 295, 297],
                                                2: [148],
                                                3: [337],
                                                4: [251],
                                                5: [283],
                                                6: [339],
                                                7: [235],
                                                8: [147],
                                                9: [284],
                                                10: [361],
                                                # 11 no match
                                                12: [292],
                                                13: [344],
                                                14: [288, 289],
                                                # 15 no match
                                                16: [381],
                                                17: [147],
                                                18: [385, 386, 101],
                                                19: [366],
                                                20: [345, 346],
                                                21: [277, 278, 279, 280],
                                                22: [348, 349],
                                                23: [150],
                                                24: [367],
                                                25: [333],
                                                26: [335],
                                                # 27 no match
                                                28: [330, 331, 332],
                                                # 29 no match
                                                # 30 no match
                                                31: [269, 270, 271, 272],
                                                32: [151],
                                                # 33 no match
                                                34: [356],
                                                35: [360],
                                                36: [346],
                                                37: [340],
                                                38: [388],
                                                # 39 no match 
                                                40: [287],
                                                41: [341],
                                                42: [291],
                                                # 43 no match
                                                44: [296], 
                                                45: [231, 232],
                                                # 46 no match
                                                # 47 no match
                                                48: [345, 346],
                                                # 49 no match
                                            }

                self.M = torch.zeros((len(self.id_to_class.keys()), 1000), dtype=torch.uint8)

                for c_AwA, cs_IN in self.mapping_AwA_ImageNet.items():
                    for c_IN in cs_IN:
                        self.M[c_AwA, c_IN] = 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        attr_tensor = self.attributes[class_id]

        attr_dict = {
            attribute_name: torch.tensor(attr_tensor[i].item())
            for i, attribute_name in enumerate(self.attribute_names)
        }

        if self.reference_ds is not None: 
            label = self.M[class_id]
        else:
            label = torch.tensor(class_id)

        sample = {
            'image': img,
            'label': label,
            **attr_dict
        }

        return sample
    
class AwA(DatasetWrap):

    def __init__(self, **kwargs): 
        self.path = kwargs.get('path')
        self.transform = kwargs.get('transform')
        self.train_ratio = kwargs.get('train_ratio', 0.8)
        self.seed = kwargs.get('seed', 42)
        self.reference_ds = kwargs.get('reference_ds', None)
        self.train_ratio = kwargs.get('train_ratio', 0.8)

        assert 0.0 < self.train_ratio < 1.0
        return

    def __load_data__(self, **kwargs):
        """
        Load and prepare CUB data.
        """
        self.__dataset__ = {}

        # Load train split
        _ds = CustomDS(
            path=self.path,
            transform=self.transform,
            reference_ds=self.reference_ds,
            seed=self.seed
        )

        self.__dataset__ = {}
        
        # split train into train and test
        self.__dataset__['train'], self.__dataset__['test'] = torch.utils.data.random_split(
                _ds,
                [1 - self.train_ratio, self.train_ratio],
                generator = torch.Generator().manual_seed(self.seed)
        )

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
    
