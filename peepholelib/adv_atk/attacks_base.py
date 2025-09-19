# general python stuff
from pathlib import Path as Path
import abc 
from tqdm import tqdm

# Torch stuff
import torch
import torchattacks
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

from peepholelib.datasets.dataset_base import DatasetBase

def ftd(data, key_list):
    r = {}
    for k in key_list:
        r[k] = data[k]
    return r 

class AttackBase(DatasetBase):
    
    def __init__(self, **kwargs):

        DatasetBase.__init__(self, **kwargs)
        self.path = Path(kwargs['path'])
        self.name = Path(kwargs['name'])
        self.mode = kwargs['mode'] if 'mode' in kwargs else None
        
        self.model = None
        self._loaders = None
        self.res = None

    
    def load_data(self, **kwargs):
        if not self.data_path.exists(): raise RuntimeError(f'Attack path {self.data_path} does not exist. Please run get_ds_attack() first.')
        print(self.data_path)
        self.__dataset__ = {}
        if self.verbose: print(f'File {self.data_path} exists.')
        for ds_key in self._loaders:
            self.__dataset__[ds_key] = TensorDict.load_memmap(self.data_path/ds_key)

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
        data = {'image': self.__dataset__[ds_key]['image'][idx].unsqueeze(0),
                'label': self.__dataset__[ds_key]['label'][idx].unsqueeze(0),
                'attack_success': self.__dataset__[ds_key]['attack_success'][idx].unsqueeze(0)}
        return data
   
    def get_ds_attack(self):
        self.data_path.mkdir(parents=True, exist_ok=True)

        attack_TensorDict = {}
        
        for loader_name in self._loaders:
            
            if self.verbose: print(f'\n ---- Getting data from {loader_name}\n')
            n_samples = len(self._loaders[loader_name].dataset)

            if self.verbose: print('loader n_samples: ', n_samples) 
            #TODO: check device
            attack_TensorDict[loader_name] = TensorDict(batch_size=n_samples) 

            file_path = self.data_path/(loader_name)
            n_threads = 32
            
            bs = self._loaders[loader_name].batch_size
            _img, _ = self._loaders[loader_name].dataset[0]
            
            attack_TensorDict[loader_name]['image'] = MMT.empty(shape=torch.Size((n_samples,)+_img.shape))
            attack_TensorDict[loader_name]['label'] = MMT.empty(shape=torch.Size((n_samples,)))
            attack_TensorDict[loader_name]['attack_success'] = MMT.empty(shape=torch.Size((n_samples,)))
            for bn, data in enumerate(tqdm(self._loaders[loader_name])):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                n_in = len(images)
                attack_images = self.atk(images, labels)
                
                with torch.no_grad():
                        y_predicted = self.model(attack_images)
                predicted_labels = y_predicted.argmax(axis = 1)
                results = predicted_labels != labels

                attack_TensorDict[loader_name]['image'][bn*bs:bn*bs+n_in] = attack_images
                attack_TensorDict[loader_name]['label'][bn*bs:bn*bs+n_in] = labels
                attack_TensorDict[loader_name]['attack_success'][bn*bs:bn*bs+n_in] = results                
            
            # if self.verbose: print(f'Saving {loader_name} to {file_path}.')
            attack_TensorDict[loader_name].memmap(file_path, num_threads=n_threads)
            self._dss = attack_TensorDict
        return
