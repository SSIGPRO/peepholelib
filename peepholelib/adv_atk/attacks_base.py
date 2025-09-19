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

def from_tensorDict(data, key_list):
    return {k: data[k] for k in key_list} 

class AttackBase(DatasetBase):
    
    def __init__(self, **kwargs):

        DatasetBase.__init__(self, **kwargs)

        self.model = kwargs.get('model')
        self.name_model = kwargs.get('name_model')

        return
    
    def load_only(self, **kwargs):

        loaders = kwargs.get('loaders')
        verbose = kwargs.get('verbose', False)

        if not self.save_path.exists(): raise RuntimeError(f'Attack path {self.save_path} does not exist. Please run get_ds_attack() first.')

        if verbose: print(f'Loading files {self.save_path} from disk. ')
        
        self._dss = {}
        for ds_key in loaders:
            self._dss[ds_key] = TensorDict.load_memmap(self.save_path/ds_key)
        return
    
    def get_ds_attack(self):
        # TODO: make documentation after rework
        '''
        Applies attacks to a dataset, saving attacked images in a `TensorDict`.

        Args:
        -
        '''
        
        _loaders = kwargs.get('loaders')
        device = kwargs.get('device') 

        self.save_path.mkdir(parents=True, exist_ok=True)
        
        attack_TensorDict = {}
        for loader_name in _loaders:
            
            if self.verbose: print(f'\n ---- Getting data from {loader_name}\n')
            n_samples = len(_loaders[loader_name].dataset)

            if self.verbose: print('loader n_samples: ', n_samples) 
            attack_TensorDict[loader_name] = TensorDict(batch_size=n_samples) 

            file_path = self.save_path/(loader_name)
            n_threads = 32
            
            bs = _loaders[loader_name].batch_size
            _img, _ = _loaders[loader_name].dataset[0]
            
            attack_TensorDict[loader_name]['image'] = MMT.empty(shape=torch.Size((n_samples,)+_img.shape))
            attack_TensorDict[loader_name]['label'] = MMT.empty(shape=torch.Size((n_samples,)))
            attack_TensorDict[loader_name]['attack_success'] = MMT.empty(shape=torch.Size((n_samples,)))
            for bn, data in enumerate(tqdm(_loaders[loader_name])):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
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

        data = {
                'image': self._dss[ds_key]['image'][idx].unsqueeze(0),
                'label': self._dss[ds_key]['label'][idx].unsqueeze(0),
                'attack_success': self._dss[ds_key]['attack_success'][idx].unsqueeze(0)}
        return data
