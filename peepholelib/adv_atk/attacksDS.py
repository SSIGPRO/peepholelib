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
from peepholelib.datasets.parsedDataset import ParsedDataset

class AttacksDS(ParsedDataset):
    
    def __init__(self, **kwargs):
        ParsedDataset.__init__(self, **kwargs)
        return
    
    def apply_attacks(self, **kwargs):
        # TODO: make documentation after rework
        '''
        Apply attacks to a dataset, saving attacked images in a `TensorDict`.

        Args:
        - dataset (peepholelib.datasets.dataset_base.DatasetBase): Parsed dataset.
        - loaders (list[str]): list of loaders in the dataset to apply the attacks to.
        - attacks (dick({str: object})): dictionary with keys being attack names and values an attack instance wrapped with  `peepholelib.adv_atk.attack_base.attackBase` inferface.
        - batch_size (int): Creates dataloader to do computation in batch size. Defaults to 64.
        - n_threads (int): 'num_workers' passed to 'torch.utils.data.DataLoader'. Defaults to 1.
        - verbose (bool): print progress messages.

        '''
        
        self.check_uncontexted()

        # input ds
        ds = kwargs.get('dataset') 
        loaders = kwargs.get('loaders', None)

        # attacks
        attacks = kwargs.get('attacks')

        # processing
        bs = kwargs.get('batch_size', 64) 
        n_threads = kwargs.get('n_threads', 1) 

        verbose = kwargs.get('verbose', False) 

        self.path.mkdir(parents=True, exist_ok=True)

        self._dss = {}
        for atk_name, atk in attacks.items():
            if verbose: print(f'\n ---- Applying {atk_name}\n')

            for ds_key in loaders:
                tdsk = atk_name + '-' + ds_key
                file_path = self.path/('dss.'+tdsk) 

                if file_path.exists():
                    if verbose: print(f'{file_path} exists. I am not overwritting it. Skipping')
                    continue

                n_samples = len(ds._dss[ds_key])
                if verbose: print(f' Got {n_samples} samples from {ds_key}')

                self._dss[tdsk] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode = 'w')

                for key in ds._dss[ds_key].keys():
                    key_sample = ds._dss[ds_key][key][0:1]
                    key_shape = key_sample.shape[1:]
                    self._dss[tdsk][key] = MMT.empty(
                        shape=torch.Size((n_samples,)+key_shape), 
                        dtype=key_sample.dtype,
                        device=key_sample.device
                        )

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
                
                keys_to_overwrite = ['image', 'output', 'pred', 'result']

                keylist = list(ds._dss[ds_key].keys())

                keys_to_copy = [k for k in keylist if k not in keys_to_overwrite]

                for di, dt in tqdm(zip(dl_ori, dl_dst), disable=not verbose, total=ceil(n_samples/bs)): 
                    ori_images = di['image'].to(atk.model.device)

                    for key in keys_to_copy:
                        dt[key] = di[key]
                    
                    atk_images = atk(
                            images = ori_images,
                            labels = di['label'].to(atk.model.device)
                            )
                    
                    with torch.no_grad():
                        y_pred_ori = atk.model(ori_images.to(atk.model.device))
                        y_pred_atk = atk.model(atk_images.to(atk.model.device))
                    pred_labels_ori = y_pred_ori.argmax(axis = 1)
                    pred_labels_atk = y_pred_atk.argmax(axis = 1)

                    dt['image'] = atk_images
                    dt['output'] = y_pred_atk
                    dt['pred'] = pred_labels_atk
                    dt['result'] = pred_labels_atk == dt['label']
                    dt['attack_success'] = torch.logical_and(pred_labels_ori == dt['label'], pred_labels_atk != pred_labels_ori) 
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
