# Our stuff
from peepholelib.datasets.dataset_base import DatasetBase
from peepholelib.datasets.transforms import vgg16_cifar10, vgg16_cifar100

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

class CifarC(DatasetBase):
    def __init__(self, **kwargs):
        '''
        CifarC loader (test).

        Expects:
            data_path (str): CifarC download folder. If not downloaded, better you download it.
        Returns:
            - a thumbs up
        '''

        DatasetBase.__init__(self, **kwargs)
        
        self.dataset = kwargs.get('dataset')

        # raise error if the dataset is not CIFAR
        if "cifar" not in self.dataset.lower():
            raise ValueError("Dataset must be CIFAR<10C|100C>")

        return
    
    def __load_data__(self, **kwargs):
        '''
        Load and prepare CIFAR10C or CIFAR100C data.
        
        Args:
        - seed (int): Random seed for reproducibility.
        - transform (torchvision.transforms.Compose): Custom transform to apply to the original dataset
        - data_path (str): Path for corrupted data (CIFAR-100-C). Saved as 'ood' loader.
        
        Returns:
        - a thumbs up
        '''
        # accepts custom transform if provided in kwargs
        transform = kwargs.get('transform')

        seed = kwargs.get('seed', 42)
            
        # set torch seed
        torch.manual_seed(seed)

        c_levels = 5
        label_file = list(self.data_path.glob('labels.npy'))[0]
        _labels = np.load(label_file).astype(int)
        _labels = _labels.reshape(c_levels, int(_labels.shape[0]/c_levels))
        n_samples = _labels.shape[1]

        # list files with different corruptions
        files = list(self.data_path.glob('[!label]*.npy'))
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

        for cl in range(c_levels):
            self.__dataset__[f'val-ood-c{cl}'] = corrupted_datasets_val[cl]
            self.__dataset__[f'test-ood-c{cl}'] = corrupted_datasets_test[cl] 
        
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
        if not self.__dataset__:
            raise RuntimeError('Data not loaded. Please run load_data() first.')
        
        return [self.__dataset__[ds_key][idx]]
