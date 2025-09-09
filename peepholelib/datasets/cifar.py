# Our stuff
from peepholelib.datasets.dataset_base import DatasetBase
from peepholelib.datasets.transforms import vgg16_cifar10, vgg16_cifar100

# General python stuff
from pathlib import Path as Path
import numpy as np
from math import floor
from tqdm import tqdm

# torch stuff
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset

# CIFAR from torchvision
from torchvision import datasets
from PIL import Image

class CustomDS(Dataset):
    def __init__(self, data, labels, transform):
        Dataset.__init__(self) 
        self.data = []
        for d in tqdm(data, disable=True):
            self.data.append(Image.fromarray(d))
        self.labels = labels
        self.transform = transform
        self.len = labels.shape[0]
        return

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        d = self.transform(self.data[idx])
        l = self.labels[idx]
        return d, l

    def __getitems__(self, idxs):
        return [(self.transform(self.data[i]), self.labels[i]) for i in idxs]

class Cifar(DatasetBase):
    def __init__(self, **kwargs):
        '''
        Cifar loader (train & val & test). Validation is created from train, fixed in 0.8 for train and 0.2 for val.

        Expects:
            data_path (str): Cifar download folder. If not downloaded, downloads the dataset in this folder.
        Returns:
            - a thumbs up
        '''

        DatasetBase.__init__(self, **kwargs)
        
        # use CIFAR10 by default
        self.dataset = kwargs.get('dataset', 'CIFAR10')

        # raise error if the dataset is not CIFAR
        if "cifar" not in self.dataset.lower():
            raise ValueError("Dataset must be CIFAR<10|100>")

        return
    
    def load_data(self, **kwargs):
        '''
        Load and prepare CIFAR10 or CIFAR100 data.
        
        Args:
        - seed (int): Random seed for reproducibility.
        - transform (torchvision.transforms.Compose): Custom transform to apply to the original dataset. (default: CIFAR10/CIFAR100 for vgg16 transform)
        - corrupted_path (str): Path for corrupted data (CIFAR-100-C). Saved as 'ood' loader.
        
        Returns:
        - a thumbs up
        '''
        # accepts custom transform if provided in kwargs
        transform = kwargs.get('transform', eval('vgg16_'+self.dataset.lower()))
        corrupted_path = kwargs.get('corrupted_path', None)
        ood_dss = kwargs.get('ood_dss', None)
        if not corrupted_path == None: corrupted_path = Path(corrupted_path)

        seed = kwargs.get('seed', 42)
            
        # set torch seed
        torch.manual_seed(seed)

        # Test dataset is loaded directly
        test_dataset = datasets.__dict__[self.dataset](
            root = self.data_path,
            train = False,
            transform = transform,
            download = True
        )
        
        # train data will be splitted into training and validation
        _train_data = datasets.__dict__[self.dataset]( 
            root = self.data_path,
            train = True,
            transform = None, #transform,
            download = True
        )
        
        train_dataset, val_dataset = random_split(
            _train_data,
            [0.8, 0.2],
            generator=torch.Generator().manual_seed(seed)
        )

        # Apply the transform 
        if transform != None:
            val_dataset.dataset.transform = transform
            train_dataset.dataset.transform = transform
        
        # Load corrupted
        if not corrupted_path == None:
            c_levels = 5
            label_file = list(corrupted_path.glob('labels.npy'))[0]
            _labels = np.load(label_file).astype(int)
            _labels = _labels.reshape(c_levels, int(_labels.shape[0]/c_levels))
            n_samples = _labels.shape[1]

            # list files with different corruptions
            files = list(corrupted_path.glob('[!label]*.npy'))
            files_val = files.copy()
            files_val.reverse()

            img_shape = np.load(files[0])[0].shape

            # pre-allocate images and labels for test
            c_images_test = np.zeros((c_levels, n_samples)+img_shape, dtype=np.uint8)
            c_images_val = c_images_test.copy()

            # we get random samples for each corruption
            idxs = torch.randperm(n_samples)
            # get spc (samples per corruption) from each corruption
            n_corruptions = len(files)
            spc = floor(n_samples/n_corruptions)

            for ci, (ft, fv) in enumerate(zip(files, files_val)):
                _data_test = np.load(ft).reshape((c_levels, n_samples)+img_shape)
                c_images_test[:, idxs[ci*spc:(ci+1)*spc]] = _data_test[:, idxs[ci*spc:(ci+1)*spc]]

                _data_val = np.load(fv).reshape((c_levels, n_samples)+img_shape)
                c_images_val[:, idxs[ci*spc:(ci+1)*spc]] = _data_val[:, idxs[ci*spc:(ci+1)*spc]]
            
            # copy remainder values (n_samples % n_corruptions*spc) 
            c_images_test[:, idxs[(ci+1)*spc:]] = _data_test[:, idxs[(ci+1)*spc:]]
            c_images_val[:, idxs[(ci+1)*spc:]] = _data_val[:, idxs[(ci+1)*spc:]]

            corrupted_datasets_test = {}
            corrupted_datasets_val = {}
            for cl in range(c_levels):
                corrupted_datasets_test[cl] = CustomDS(
                        data = c_images_test[cl],
                        labels = _labels[cl],
                        transform = transform,
                        )
                
                corrupted_datasets_val[cl] = CustomDS(
                        data = c_images_val[cl],
                        labels = _labels[cl],
                        transform = transform,
                        )

        if not ood_dss == None:

            ood_dataset_val = {}
            ood_dataset_test = {}            

            for ood in ood_dss:

                if ood == 'SVHN':
                    ood_path = str(self.data_path).replace(self.dataset, "")+ood

                    # Test dataset is loaded directly
                    _test_data = datasets.__dict__[ood](
                        root = ood_path,
                        split = 'test',
                        transform = None, #transform,
                        download = True
                    )

                    _, _test_dataset = random_split(
                        _test_data,
                        [0.61585, 0.38415],
                        generator=torch.Generator().manual_seed(seed)
                    )
                    
                    # train data will be splitted into training and validation
                    _train_data = datasets.__dict__[ood]( 
                        root = ood_path,
                        split = 'train',
                        transform = None, #transform,
                        download = True
                    )
                    
                    _, _val_dataset = random_split(
                        _train_data,
                        [0.86349, 0.13651],
                        generator=torch.Generator().manual_seed(seed)
                    )

                    # Apply the transform 
                    if transform != None:
                        _val_dataset.dataset.transform = transform
                        _test_dataset.dataset.transform = transform

                elif ood == 'Places365':

                    ood_path = str(self.data_path).replace(self.dataset, "")+ood

                    _dataset = datasets.__dict__[ood]( 
                        root = ood_path,
                        split = 'val',
                        transform = None, #transform,
                        small = True,
                        download = True
                    )
                    
                    _ , _val_dataset, _test_dataset = random_split(
                        _dataset,
                        [0.4520548, 0.2739726, 0.2739726], # to get exactly 10000 samples
                        generator=torch.Generator().manual_seed(seed)
                    )

                    # Apply the transform 
                    if transform != None:
                        _val_dataset.dataset.transform = transform
                        _test_dataset.dataset.transform = transform

                elif ood == 'DTD':

                    ood_path = str(self.data_path).replace(self.dataset, "")+ood

                    _val_dataset = datasets.__dict__[ood]( 
                        root = ood_path,
                        split = 'val',
                        transform = transform,
                        download = True
                    )

                    _test_dataset = datasets.__dict__[ood]( 
                        root = ood_path,
                        split = 'test',
                        transform = transform,
                        download = True
                    )
                    
                else:
                    raise RuntimeError(f'{ood} is not a supported ood dataset supported for CIFAR')                    

                ood_dataset_val[ood] = _val_dataset
                ood_dataset_test[ood] = _test_dataset    
    
        # Save datasets as objects in the class
        self._dss = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
                }

        if not corrupted_path == None:
            for cl in range(c_levels):
                self._dss[f'val-ood-c{cl}'] = corrupted_datasets_val[cl]
                self._dss[f'test-ood-c{cl}'] = corrupted_datasets_test[cl]

        if not ood_dss == None:
            for ds in ood_dataset_val:
                self._dss[f'val-ood-{ds}'] = ood_dataset_val[ds]
                self._dss[f'test-ood-{ds}'] = ood_dataset_test[ds]

        self._classes = {i: class_name for i, class_name in enumerate(test_dataset.classes)}  
        
        return 
    
    def get(self, ds_key, idx):
        '''
        Get item from the dataset.
        
        Args:
        - idx (int): Index of the item to get.
        - ds_key (str): Key of the dataset to get the item from ('train', 'val', 'test').
        
        Returns:
        - a tuple of (image, label)
        '''
        if not self._dss:
            raise RuntimeError('Data not loaded. Please run load_data() first.')
        
        return [self._dss[ds_key][idx]]
