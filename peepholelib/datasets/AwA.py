# general python stuff
import os
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from types import NoneType
from math import floor, ceil
import json
import re

# Our stuff
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.functional.transforms import vgg16_cifar100
from peepholelib.models.prediction_fns import multilabel_classification

# torch stuff
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT

# CIFAR from torchvision
from torchvision import datasets

def onehot_to_index(bits):
    for i, b in enumerate(bits):
        if b == 1:
            return torch.tensor([i])
    return torch.tensor([0])

from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torch
import os

class CustomDS(Dataset):
    def __init__(self, path, reference_ds=None , train=True, train_ratio=0.8, seed=42, transform=None):
        """
        path: path to AwA2 folder containing JPEGImages/, classes.txt, predicate-matrix-binary.txt
        split: "train" or "test"
        train_ratio: fraction of samples used for training
        seed: for deterministic split
        """
        
        assert 0.0 < train_ratio < 1.0

        self.path = path
        self.reference_ds = reference_ds
        self.is_train = train
        self.transform = transform

        # ---- Load class names ----
        classes_file = os.path.join(path, "classes.txt")
        self.id_to_class = {}
        with open(classes_file) as f:
            for line in f:
                cid, cname = line.strip().split('\t')
                self.id_to_class[int(cid)] = cname

        # ---- Load attribute matrix ----
        attr_file = os.path.join(path, "predicate-matrix-binary.txt")
        attr_list = []
        with open(attr_file, "r") as f:
            for line in f:
                attr_list.append([float(x) for x in line.strip().split()])
        self.attributes = torch.tensor(attr_list, dtype=torch.float32)  # [50, 85]

        # ---- Load attribute names ----
        pred_file = os.path.join(path, "predicates.txt")
        self.attribute_names = []
        with open(pred_file, "r") as f:
            for line in f:
                _, name = line.strip().split('\t')
                self.attribute_names.append(name)

        assert len(self.attribute_names) == self.attributes.shape[1]

        # ---- Build full list of samples (image_path, class_id) ----
        all_samples = []
        img_path = os.path.join(path, "JPEGImages")

        for cid, cname in self.id_to_class.items():
            class_dir = os.path.join(img_path, cname)
            if not os.path.isdir(class_dir):
                print(f"WARNING: folder missing → {class_dir}")
                continue

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_samples.append((os.path.join(class_dir, fname), cid - 1))
        
        # ---- Reference ds classes names ----

        self.dict_mapping = {
            0: [351, 352, 353],
            1: [294, 295, 297],
            2: [148],
            3: [337],
            4: [251],
            5: [283],
            # 6 no match
            7: [235],
            8: [147],
            9: [284],
            10: [361],
            #11 no match
            12: [292],
            13: [344],
            14: [288,289],
            # 15 no match
            16: [381],
            17: [147],
            18: [385, 386],
            19: [366],
            20: [345, 346],
            21: [277, 278, 279, 280],
            22: [349],
            # 23 no match
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
            # 40 no match
            41: [341],
            42: [291],
            43: [673],
            44: [296], 
            45: [231, 232],
            # 46 no match
            # 47 no match
            # 48 no match
            # 49 no match

        }
        if self.reference_ds is not None:

            with open(self.reference_ds, "r") as f:
                data = json.load(f)
            
            self.reference_classes = {int(k): (v[0], v[1]) for k, v in data.items()}

            self.M = torch.zeros((len(self.id_to_class.keys()), len(self.reference_classes.keys())), dtype=torch.uint8)

            

        # ---- Deterministic train/test split ----
        n_total = len(all_samples)
        n_train = int(train_ratio * n_total)

        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=g)

        train_idx = perm[:n_train].tolist()
        test_idx  = perm[n_train:].tolist()

        if self.is_train:
            self.samples = [all_samples[i] for i in train_idx]
        else:
            self.samples = [all_samples[i] for i in test_idx]

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
    
class AwAWrap(DatasetWrap):

    def __init__(self, **kwargs): 
        self.path = kwargs.get('path')
        self.transform = kwargs.get('transform')
        self.train_ratio = kwargs.get('train_ratio', 0.8)
        self.seed = kwargs.get('seed', 42)
        self.reference_ds = kwargs.get('reference_ds', None)
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
            seed=self.seed,
            transform=self.transform,
            reference_ds=self.reference_ds
        )

        # Load test split
        test_dataset = CustomDS(
            path=self.path,
            train=False,
            seed=self.seed,
            transform=self.transform,
            reference_ds=self.reference_ds
        )

        self.__dataset__ = {
            "train": train_dataset,
            "test": test_dataset
        }

        # Build class names dictionary
        # classes = train_dataset.class_id_to_name
        # self._classes = {
        #     "CUB-train": classes,
        #     "CUB-test": classes
        # }

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
    
class AwA(ParsedDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
    
    def create_ds(self, **kwargs):
        path = Path(kwargs['path'])
        awa_wrap = kwargs['awa_wrap']
        bs = kwargs.get('batch_size', 2**11)
        verbose = kwargs.get('verbose', False)

        awa_wrap.__load_data__()
        
        path.mkdir(parents=True, exist_ok=True)
        self._dss = {}
        
        for ds_key in awa_wrap.__dataset__:
            file_path = self.path/('dss.'+ds_key)
            n_samples = len(awa_wrap.__dataset__[ds_key])
            
            # check if PTD exists 
            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                self._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            else:
                if verbose: print(f'Creating {ds_key} dataset with n_samples: ', n_samples)
                self._dss[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
                
                # get sample to get shapes
                sample = awa_wrap.__dataset__[ds_key][0]

                for key in sample.keys():
                    
                    if verbose: print(f'allocating {key} with shape {sample[key].shape}')
                    self._dss[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,)+sample[key].shape), dtype=torch.float32)
        
                # create Dataloader of input dataset
                ds_in = DataLoader(
                    dataset = awa_wrap.__dataset__[ds_key],
                    batch_size = bs
                )
                                                                                                                                        
                ds_t = DataLoader(
                    self._dss[ds_key],
                    collate_fn = lambda x:x, 
                    batch_size = bs
                )
                                                                                                                                        
                for data_in, data_t in tqdm(zip(ds_in, ds_t), disable=not verbose, total=ceil(n_samples/bs)):
                    for key in data_in.keys():
                        data_t[key] = data_in[key]
            
            # close the PTD
            self._dss[ds_key].close()
        return

    # overwrite the parse_ds() from peepholelib.datasets.parsedDataset.ParsedDataset
    def parse_ds(self, **kwargs):
        self.check_uncontexted()

        model = kwargs['model']
        loaders = kwargs.get('loaders', None)
        bs = kwargs.get('batch_size', 2**11)
        verbose = kwargs.get('verbose', False)
        pred_fn = kwargs.get('pred_fn', multilabel_classification)
        
        if loaders == None:
            loaders = self._dss.keys()

        for ds_key in loaders:
            file_path = self.path/('dss.'+ds_key)
            n_samples = len(self._dss[ds_key])
            
            # check if PTD exists 
            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                self._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
            else:
                if verbose: raise RuntimeError(f'Not {ds_key} dataset run create_ds first')

            # if ('output' in self._dss[ds_key]) and ('pred' in self._dss[ds_key]) and ('result' in self._dss[ds_key]):
            #     if verbose: print(f'keys: output, pred and result already parsed for {ds_key}. check the next')
            #     continue
            
            # dataset sample for dry run
            sample = self._dss[ds_key][0:1]['image'].to(model.device)
            with torch.no_grad():
                _out = model(sample)
                     
            os = _out.shape[1:]

            # need to fix the batch size - workaround  
            self._dss[ds_key].batch_size = torch.Size((n_samples,))
            # allocate disk space
            self._dss[ds_key]['output'] = MMT.empty(shape=torch.Size((n_samples,)+os), dtype=torch.float32)
            self._dss[ds_key]['pred'] = MMT.empty(shape=torch.Size((n_samples,)))
            self._dss[ds_key]['result'] = MMT.empty(shape=torch.Size((n_samples,)))
            print(self._dss[ds_key]['pred'].shape, self._dss[ds_key]['label'].shape)

            dl = DataLoader(
                self._dss[ds_key],
                collate_fn = lambda x:x, 
                batch_size = bs
            )

            for data in tqdm(dl, disable=not verbose, total=ceil(n_samples/bs)):
                #compute predictions which is the out of decoder

                with torch.no_grad():
                    y_predicted = model(data['image'].to(model.device))
            
                    predicted_labels = pred_fn(y_predicted).detach().cpu()
                    data['output'] = y_predicted
                    data['pred'] = predicted_labels
                    data['result'] = data['label'].gather(
                                                        dim=1,
                                                        index=predicted_labels.unsqueeze(1)
                                                    ).squeeze(1).bool()
        
        return
