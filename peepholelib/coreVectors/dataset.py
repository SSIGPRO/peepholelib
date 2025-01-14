# General python stuff
from tqdm import tqdm

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

def fds(data, key):
    if key == 'image':
        return data[0]
    if key == 'label':
        return data[1]

def get_coreVec_dataset(self, **kwargs):
    self.check_uncontexted()

    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    loaders = kwargs['loaders']
    key_list = kwargs['key_list'] if 'key_list' in kwargs else ['image', 'label']
    parser = kwargs['parser'] if 'parser' in kwargs else fds

    for ds_key in loaders:
        if verbose: print(f'\n ---- Getting data from {ds_key}\n')
        file_path = self.path/(self.name.name+'.'+ds_key)
        self._cvs_file_paths[ds_key] = file_path
        bs = loaders[ds_key].batch_size
        
        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._corevds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            n_samples = len(self._corevds[ds_key])
            if verbose: print('loaded n_samples: ', n_samples)
        else:
            n_samples = len(loaders[ds_key].dataset)
            if verbose: print('loader n_samples: ', n_samples) 
            self._corevds[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
            
            #------------------------
            # copy images and labels
            #------------------------

            if verbose: print('Allocating data')

            for key in key_list:
                data = parser(next(iter(loaders[ds_key])), key)[0]
                
                # get shapes for pre-allocation
                if data.shape == torch.Size([]):
                    self._corevds[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,))) 
                else:
                    self._corevds[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,)+data.shape))

            if verbose: print('Copying images and labels')
            dl_in = loaders[ds_key]
            dl_t = DataLoader(self._corevds[ds_key], batch_size=bs, collate_fn=lambda x:x)
            
            for data_in, data_t in tqdm(zip(dl_in, dl_t), disable=not verbose, total=len(dl_in)): 
                for key in key_list:
                    data_t[key] = parser(data_in, key)

        self._n_samples[ds_key] = n_samples

    return
