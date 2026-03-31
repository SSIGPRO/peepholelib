# Our stuff
from peepholelib.datasets.datasetWrap import DatasetWrap

# torch stuff
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor 

# CIFAR from torchvision
from PIL import Image
import numpy as np
from math import floor

class CustomDS(Dataset):
    def __init__(self, data, labels, corruptions):
        Dataset.__init__(self) 
        self._to_tensor = ToTensor()

        p = [
            'defocus_blur', 
            'glass_blur', 
            'spatter', 
            'contrast', 
            'fog', 
            'speckle_noise', 
            'shot_noise', 
            'impulse_noise', 
            'pixelate', 
            'snow', 
            'zoom_blur', 
            'gaussian_blur', 
            'jpeg_compression', 
            'gaussian_noise', 
            'frost', 
            'saturate', 
            'motion_blur'
            ]

        self.mapping = {c: i for i, c in enumerate(p)}
        
        self.data = []
        for d in data:
            self.data.append(Image.fromarray(d))
        self.labels = labels

        name_corruptions = set(corruptions)
    
        unknown = sorted(name_corruptions - set(self.mapping.keys()))
        if unknown:
            raise ValueError(f'Unknown corruption names found: {unknown}')
        self.corruptions = torch.tensor([self.mapping[c] for c in corruptions], dtype=torch.long)
        self.len = labels.shape[0]
        return

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {
            "image": self._to_tensor(self.data[idx]),
            "label": self.labels[idx],
            "corruption": self.corruptions[idx],
        }
        return sample

class CifarC(DatasetWrap):
    def __init__(self, **kwargs):
        '''
        CIFAR-C loader (val & test). Builds corruption-level datasets from precomputed `.npy` corruption files and `labels.npy`.

        Args:
            path (str): Path to the CIFAR-C folder containing `labels.npy` and corruption `.npy` files.
            seed (int, optional): Random seed used to sample corruption mixes reproducibly.
        Returns:
            - a thumbs up
        '''

        DatasetWrap.__init__(self, **kwargs)
        
        return
    
    def __load_data__(self):
        '''
        Load and prepare CIFAR10-C or CIFAR100-C data.
        
        Returns:
        - a thumbs up
        '''
            
        # set torch seed
        torch.manual_seed(self.seed)

        c_levels = 5
        label_file = list(self.path.glob('labels.npy'))[0]

        _labels = torch.from_numpy(np.load(label_file)).long()
        _labels = _labels.view(c_levels, _labels.numel() // c_levels)
        n_samples = _labels.shape[1]

        # list files with different corruptions
        files = list(self.path.glob('[!label]*.npy'))
        files_val = files.copy()
        files_val.reverse()

        img_shape = np.load(files[0])[0].shape

        # pre-allocate images and labels for test
        c_images_test = np.zeros((c_levels, n_samples)+img_shape, dtype=np.uint8)
        c_images_val = c_images_test.copy()
        c_corruptions_test = np.empty(n_samples, dtype=object)
        c_corruptions_val = np.empty(n_samples, dtype=object)

        # we get random samples for each corruption
        idxs = torch.randperm(n_samples)
        # get spc (samples per corruption) from each corruption
        n_corruptions = len(files)
        spc = floor(n_samples/n_corruptions)

        for ci, (ft, fv) in enumerate(zip(files, files_val)):
            _data_test = np.load(ft).reshape((c_levels, n_samples)+img_shape)
            dst_idxs = idxs[ci*spc:(ci+1)*spc]
            c_images_test[:, dst_idxs] = _data_test[:, dst_idxs]
            c_corruptions_test[dst_idxs] = ft.stem

            _data_val = np.load(fv).reshape((c_levels, n_samples)+img_shape)
            c_images_val[:, dst_idxs] = _data_val[:, dst_idxs]
            c_corruptions_val[dst_idxs] = fv.stem
        
        # copy remainder values (n_samples % n_corruptions*spc) 
        rem_idxs = idxs[(ci+1)*spc:]
        c_images_test[:, rem_idxs] = _data_test[:, rem_idxs]
        c_images_val[:, rem_idxs] = _data_val[:, rem_idxs]
        c_corruptions_test[rem_idxs] = ft.stem
        c_corruptions_val[rem_idxs] = fv.stem

        corrupted_datasets_test = {}
        corrupted_datasets_val = {}
        for cl in range(c_levels):
            corrupted_datasets_test[cl] = CustomDS(
                    data = c_images_test[cl],
                    labels = _labels[cl],
                    corruptions = c_corruptions_test,
                    )
            
            corrupted_datasets_val[cl] = CustomDS(
                    data = c_images_val[cl],
                    labels = _labels[cl],
                    corruptions = c_corruptions_val,
                    ) 

        self.__dataset__ = {}
        for cl in range(c_levels):
            self.__dataset__[f'CIFAR100-C-val-c{cl}'] = corrupted_datasets_val[cl]
            self.__dataset__[f'CIFAR100-C-test-c{cl}'] = corrupted_datasets_test[cl] 
        
        return 
    
