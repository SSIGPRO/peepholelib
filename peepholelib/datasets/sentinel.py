
# general python stuff
from pathlib import Path
import pandas as pd

# torch stuff
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# peepholelib imports
from peepholelib.datasets.dataset_base import DatasetBase
from peepholelib.datasets.transforms import vgg16_imagenet

class Sentinel(DatasetBase):#DatasetBase

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #DatasetBase.__init__(self, **kwargs)
        
        #self.freq = kwargs.get('freq', '4s')
        
        return

    def load_data(self, **kwargs):
        train_file = Path(self.data_path)/'train_data.pkl'
        test_file = Path(self.data_path)/'test_data.pkl'

        data_train_std = pd.read_pickle(train_file.as_posix())
        data_test_std = pd.read_pickle(test_file.as_posix())
        #print(f'Data_Train:{data_train_std}\nData_Test:{data_test_std}')
        #data_test_std = pd.read_pickle((self.data_dir, self.dataset, 'test_data.pkl'))
        

        seed = kwargs['seed']
        #data_test_std['label'] = 0
        data_train_df_ = data_train_std.values
        train_set , val_set = torch.utils.data.random_split(
                data_train_df_,
                [0.8, 0.2],
                generator = torch.Generator().manual_seed(seed)
        )
        print(type(train_set))
        print(f'train set:{train_set}\nval set:{val_set}')
        quit()
        '''
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