# general python stuff
from pathlib import Path as Path
import abc 
from tqdm import tqdm
from math import ceil

# tensordict
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT

# Torch stuff
import torch
import torchattacks
from torch.utils.data import DataLoader

# our stuff
from peepholelib.datasets.dataset_base import DatasetBase

def from_tensorDict(data, key_list):
    return {k: data[k] for k in key_list} 

class AttackBase(DatasetBase):
    
    def __init__(self, **kwargs):

        DatasetBase.__init__(self, **kwargs)

        self.model = kwargs.get('model')
        return
    
    def apply_attack(self, **kwargs):
        # TODO: make documentation after rework
        '''
        Applies attacks to a dataset, saving attacked images in a `TensorDict`.

        Args:
        -
        '''
        
        self.check_uncontexted()

        ds = kwargs.get('dataset') 
        loaders = kwargs.get('loaders', None)
        bs = kwargs.get('batch_size', 64) 
        n_threads = kwargs.get('n_threads', 1) 

        self.path.mkdir(parents=True, exist_ok=True)
        verbose = kwargs.get('verbose', False) 

        # some defs for simplicity 
        device = self.model.device
        
        self._dss = {}
        for ds_key in loaders:
            tdsk = self.name + ds_key
            file_path = self.path/('dss.'+tdsk) 

            if file_path.exists():
                print(f'{file_path} exists. I am not overwritting it. Skipping')
                continue

            if verbose: print(f'\n ---- Getting data from {ds_key}\n')
            n_samples = len(ds._dss[ds_key])
            if verbose: print('loader n_samples: ', n_samples) 

            self._dss[tdsk] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode = 'w') 

            img_shape = ds._dss[ds_key]['image'][0].shape
            
            self._dss[tdsk]['image'] = MMT.empty(shape=torch.Size((n_samples,)+img_shape))
            self._dss[tdsk]['label'] = MMT.empty(shape=torch.Size((n_samples,)))
            self._dss[tdsk]['attack_success'] = MMT.empty(shape=torch.Size((n_samples,)))

            # Close PTD create with mode 'w' and re-open it with mode 'r+'
            # This is done so we can use multiple workers with the dataloaders 
            self._dss[tdsk].close()
            self._dss[tdsk] = PersistentTensorDict.from_h5(file_path, mode='r+')
             
            #------------------------
            # copy images and labels
            #------------------------               
            # create dataloader of input dataset
            dl_ori = DataLoader(
                    dataset = ds._dss[ds_key],
                    batch_size = bs,
                    collate_fn = lambda x:x,
                    shuffle = False
                    ) 
                                                                
            dl_dst = DataLoader(
                    self._dss[tdsk],
                    batch_size = bs,
                    collate_fn = lambda x:x,
                    shuffle = False,
                    num_workers = n_threads
                    )

            if verbose: print(f'Applying {self.name} to {ds_key}')
            for di, dt in tqdm(zip(dl_ori, dl_dst), disable=not verbose, total=ceil(n_samples/bs)): 

                images = di['image'].to(device)
                labels = di['label'].to(device)
                attack_images = self.atk(images, labels)
                
                with torch.no_grad():
                    y_predicted = self.model(attack_images)
                predicted_labels = y_predicted.argmax(axis = 1)
                results = predicted_labels != labels

                dt['image'] = attack_images
                dt['label'] = labels
                dt['attack_success'] = results                
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
