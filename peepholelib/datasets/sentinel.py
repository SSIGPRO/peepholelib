
# general python stuff
from pathlib import Path
from types import NoneType
import pandas as pd
from math import ceil, floor 
import numpy as np
from tqdm import tqdm
from cuda_selector import auto_cuda
import abc

# tensordict
from peepholelib.datasets import parsedDataset
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.models.prediction_fns import multilabel_classification
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT

# torch stuff
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

class CustomDS(Dataset):
    def __init__(self, data, labels):
        Dataset.__init__(self) 
        ws = 16 # window size

        _data = torch.tensor(data.values, dtype=torch.float32)
        nw = floor(_data.shape[0]/ws) # num windows
        print('\ndata shape before trimming: ', _data.shape)
        data = _data[:ws*nw]
        print('data shape  after trimming: ', data.shape)
        data = data.reshape(-1, ws, data.shape[-1])#.float() # 16 is the number of signals
        print('data shape  after reshaping: ', data.shape)

        # TODO: do the NaN check by batch, not before batch
        idx = data.isnan().any(dim=(1,2)).logical_not()#used dim instead of axis
        self.data = data[idx]#.float()
        print('data shape  after dropping: ', self.data.shape)

        if type(labels) == NoneType:
            self.labels = labels  
        elif type(labels) == pd.DataFrame:
            _labels = torch.tensor(labels.values, dtype=torch.float32)
            print('labels shape before trimming: ', _labels.shape)
            labels = _labels[:ws*nw]
            print('labels shape  after trimming: ', labels.shape)
            labels = labels.reshape(-1, ws, labels.shape[-1])
            print('labels shape  after reshaping: ', labels.shape)
            self.labels = labels[idx]
            print('labels shape  after dropping: ', self.labels.shape)
            
            #self.labels = self.labels.reshape(-1, window_size, window_size)
            #self.labels = torch.tensor(labels.values, dtype=torch.float32)[idx]
        else: 
            raise RuntimeError('Dude, wtf?!')
        
        self.len = self.data.shape[0]
        return

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.labels == None:
            return {'data': self.data[idx]}
        else:
            return {'data': self.data[idx], 'label': self.labels[idx]}
        
class SentinelWrap(DatasetWrap):#DatasetBase

    def __init__(self, **kwargs):
        
        self.path = kwargs.get('path')
        self.train_val_split = kwargs.get('train_val_split', 0.2)
        
        return

    def __load_data__(self, **kwargs):
        
        # load data and the labels
        train_file = Path(self.path)/'train_data.pkl'
        test_file = Path(self.path)/'test_data.pkl'
        label_file = Path(self.path)/'test_labels.pkl'

        data_train_std = pd.read_pickle(train_file.as_posix())
        data_test_std = pd.read_pickle(test_file.as_posix())
        test_labels = pd.read_pickle(label_file.as_posix())

        print(f'test_labels.shape={test_labels.shape}')
        seed = kwargs.get('seed', 42)
        verbose = kwargs.get('verbose', False)

        torch.manual_seed(seed)
        self.__dataset__ = {}
        
        # split train into train and val
        ds_train = CustomDS(data=data_train_std, labels=None)
        self.__dataset__['train'] , self.__dataset__['val'] = torch.utils.data.random_split(
                ds_train,
                [1-self.train_val_split, self.train_val_split],
                generator = torch.Generator().manual_seed(seed)
        )
        # test
        self.__dataset__['test'] = CustomDS(data=data_test_std, labels=test_labels)
        #print(f'Hakuna Matata:{self.__dataset__['train'][0]['data'].dtype}')

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
    
class Sentinel(parsedDataset.ParsedDataset):
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            return
        @classmethod
        def parse_ds(cls, **kwargs):
            model_wrap = kwargs.get('model')
            verbose = kwargs.get('verbose')
            parsed_path = Path(kwargs.get('parsed_path'))
            sentinel_wrap = kwargs.get('sentinel_wrap')
            parsed_path.mkdir(parents=True, exist_ok=True)
            cls_inst = cls(path = parsed_path)
            cls_inst._dss = {}
            
            for ds_key in sentinel_wrap.__dataset__:
                
                file_path = cls_inst.path/('dss.'+ds_key)
                n_samples = len(sentinel_wrap.__dataset__[ds_key])
                
                # dataset sample ps:ds_smaple is a dict
                ds_sample = sentinel_wrap.__dataset__[ds_key][0]
                model_input = ds_sample['data'].unsqueeze(0)
                
                
                with torch.no_grad():
                    _output, latent_space = model_wrap(model_input.to(model_wrap.device))                
                os = _output.shape[1:]
                ls = latent_space.shape[1:]
                if verbose: print(f'output shape{os}')
                
                
                # check if PTD exists ps:PTD stands for PersistentTensorDict
                if file_path.exists():
                    if verbose: print(f'File {file_path} exists. Loading from disk.')
                    cls_inst._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
                   
       
                    # create a new one if not
                else:
                    if verbose: print('Creating dataset with n_samples: ', n_samples)
                    cls_inst._dss[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
            
                    # allocate disk space
                    cls_inst._dss[ds_key]['output'] = MMT.empty(shape=torch.Size((n_samples,)+os), dtype=torch.float32)
                    cls_inst._dss[ds_key]['latent_space'] = MMT.empty(shape=torch.Size((n_samples,)+ls), dtype=torch.float32)

                    # Close PTD create with mode 'w' and re-open it with mode 'r+'
                    # This is done so we can use multiple workers with the dataloaders 
                    cls_inst._dss[ds_key].close()
                    cls_inst._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')


                    # create Dataloader of input dataset
                    sentinel_origin = DataLoader(
                        dataset = sentinel_wrap.__dataset__[ds_key],
                        batch_size = 16
                    )

                    sentinel_destination = DataLoader(
                        cls_inst._dss[ds_key],
                        collate_fn = lambda x:x, 
                        batch_size = 16
                    )

                    #if verbose: print(f'Parsing {ds_key}')
                
                    for data_in, data_t in tqdm(zip(sentinel_origin, sentinel_destination), disable=not verbose, total=n_samples/16): 
                        for key in data_in.keys():
                                
                            data_t[key] = data_in[key]
                #print(f'data_t{data_t}')

                    #compute predictions which is the out of decoder
                    with torch.no_grad():
                        y_predicted, latent = model_wrap(data_t['data'].float().to(model_wrap.device))
                    
                        
                        predicted_labels = y_predicted.detach().cpu()
                    
                        data_t['output'] = y_predicted
                        data_t['latent_space'] = latent            
            
            return
        
        
        def load_only(self, **kwargs):
            '''
        Load already computed dataset.

        Args:
        - loadrs (list[str]): load the specified loaders
        - mode (str): Opens the file with the specified mode. See 'tensordict.PersistentTensorDict.from_h5()' for details. Defaults to 'r'.
        - verbose (bool): print progress messages.
        '''
            self.check_uncontexted()

            loaders = kwargs.get('loaders')
            mode = kwargs.get('mode', 'r')
            verbose = kwargs.get('verbose', False)

            self._dss = {}
            for ds_key in loaders:
                if verbose: print(f'\n ---- Getting data from {ds_key}\n')
            
                # data file path is the one which was converted to dss above
                _dfp = self.path/('dss.'+ds_key)

                if verbose: print(f'Loading files {_dfp} from disk. ')
                self._dss[ds_key] = PersistentTensorDict.from_h5(_dfp, mode=mode)
                #print(f'Loaded from h5:{self._dss[ds_key]['data'].dtype}') #i get NONE ?!!!!
                _n_samples = len(self._dss[ds_key])
                if verbose: print('loaded n_samples: ', _n_samples)
                

            return
