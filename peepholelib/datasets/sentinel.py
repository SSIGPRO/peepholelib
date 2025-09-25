
# general python stuff
from pathlib import Path
from types import NoneType
import pandas as pd
import math 
import numpy as np
from cuda_selector import auto_cuda
# tensordict
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT

# torch stuff
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

# peepholelib imports
from peepholelib.datasets.dataset_base import DatasetBase

class CustomDS(Dataset):
    def __init__(self, data, labels=None):
        Dataset.__init__(self) 
        _data = torch.tensor(data.values)
        idx = _data.isnan().any(axis=1).logical_not()
        self.data = _data[idx]
        
        if type(labels) == NoneType:
            self.labels = labels    
        elif type(labels) == pd.DataFrame:
            self.labels = torch.tensor(labels.values)[idx]
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
        
class Sentinel(DatasetBase):#DatasetBase

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #DatasetBase.__init__(self, **kwargs)

        # added these cuz get error regarding the device location before
        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")

        self.parsed_path = Path(kwargs.get('parsed_path'))
        self.train_val_split = kwargs.get('train_val_split', 0.2)
        self.model = kwargs.get('model')
        
        #Breaking here when try this way to fix Breaking Point 1
        #ERROR: 'CONV_AE1D' object has no attribute 'seek'. You can only torch.load from a file that is seekable.
        #Please pre-load the data into a buffer like io.BytesIO and try to load from it instead.

        #checkpoint_path = kwargs.get('model_path')
        #self.model.load_state_dict(torch.load(self.model, map_location=device))
        #self.model.to(device) 
        # the line below didn't work
        #device = next(self.model.parameters()).device

        return

    def load_data(self, **kwargs):
        # load data and the labels
        train_file = Path(self.path)/'train_data.pkl'
        test_file = Path(self.path)/'test_data.pkl'
        label_file = Path(self.path)/'test_labels.pkl'

        data_train_std = pd.read_pickle(train_file.as_posix())
        data_test_std = pd.read_pickle(test_file.as_posix())
        test_labels = pd.read_pickle(label_file.as_posix())

        seed = kwargs.get('seed', 42)
        verbose = kwargs.get('verbose', False)

        torch.manual_seed(seed)

        datasets = {}

        # split train into train and val
        ds_train = CustomDS(data=data_train_std, labels=None)
        datasets['train'] , datasets['val'] = torch.utils.data.random_split(
                ds_train,
                [1-self.train_val_split, self.train_val_split],
                generator = torch.Generator().manual_seed(seed)
        )
        # test
        datasets['test'] = CustomDS(data=data_test_std, labels=test_labels)
              

        for ds_key in datasets:
            
            file_path = self.parsed_path/('dss.'+ds_key)
            n_samples = len(datasets[ds_key])

            if verbose: print(f'Parsing {ds_key} with {n_samples} samples.')
            
            # dataset sample
            ds = datasets[ds_key][0]
            if verbose: print(f'{ds}')
    
            # dry run to get output shape
            _sample = datasets[ds_key][0]
            #if verbose: print(f'checking what is _samples{_sample}\n{type(_sample)}')

            # when do print(x.shape) i get tensor size[16] and get error so found online this solution .unsqueeze()
            # .unsqueeze(0) -> [1,16] .unsqueeze(-1) -> [1,16,1] also need .float cuz was float64 the sample &expected float32
            model_input = _sample['data'].unsqueeze(0).unsqueeze(-1).float()
            #print(f'model_input{model_input}\nShape of it{model_input.shape}')
            
            
            with torch.no_grad():
                #Breaking Point 1:
                #RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! 
                #(when checking argument for argument weight in method wrapper_CUDA___slow_conv2d_forward)
                _output = self.model(model_input)
                #if verbose: print(f'output shape{_output}')
            os = _output.shape
            if verbose: print(f'output shape{os}')
            continue

            '''
            # check if PTD exists
            if file_path.exists():
                #if verbose: print(f'File {file_path} exists. Loading from disk.')
                datasets[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
       
            # create a new one if not
            else:
                #if verbose: print('Creating dataset with n_samples: ', n_samples)
                datasets[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
            
            # allocate disk space
            datasets[ds_key]['output'] = MMT.empty(shape=torch.Size((n_samples,)))
            datasets[ds_key]['pred'] = MMT.empty(shape=torch.Size((n_samples,)))
            datasets[ds_key]['result'] = MMT.empty(shape=torch.Size((n_samples,)))

            # Close PTD create with mode 'w' and re-open it with mode 'r+'
            # This is done so we can use multiple workers with the dataloaders 
            datasets[ds_key].close()
            datasets[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')


            #print(type(train_set))
            #print(f'train set:{train_set}\nval set:{val_set}')
            #quit()
            
            #transform = kwargs['transform'] if 'transform' in kwargs else ToTensor()

            #seed = kwargs['seed']
            #torch.manual_seed(seed)

            data_train_df = pd.DataFrame(data_train_std)
            data_test_df = pd.DataFrame(data_test_std)
            # print(f"Train data as DataFrame:\n{data_train_df}, \nTest data as DataFrame:\n{data_test_df}")
            
            # dummy labels needed cua after split it expects (data,label) data->is subset actually
            data_train_df['label'] = 0
            data_test_df['label']  = 0

            # Convert DataFrame to tensor
            data_train_df_ = data_train_df.values
            
            # set torch seed
            seed = 42
            gen = torch.Generator()    
            torch.manual_seed(seed)

            # split the train set in train and validation set
            train_set , val_set = torch.utils.data.random_split(
                    data_train_df_,
                    [0.8, 0.2],
                    generator = gen
            )

            # convert again to dataframe
            train_df = data_train_df.iloc[train_set.indices].reset_index(drop=True)
            val_df   = data_train_df.iloc[val_set.indices].reset_index(drop=True)
            # Print to check
            print(f"Test data as DataFrame:\n{data_test_df},\nTrain data as DataFrame:\n{train_df},\nValidation data:\n{val_df}")

            # metadata
            #self._dss = {"train": train_df, "val": val_df}
            #self._classes = {i: c for i, c in enumerate(train_ds.classes)}
            '''
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