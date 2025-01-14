# Our stuff
from datasets.dataset_base import DatasetBase

# torch stuff
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset


class Dummy(DatasetBase):
    # costruttore
    def __init__(self, **kwargs):
        DatasetBase.__init__(self, **kwargs)

		# # se passato in input la dim del dataset - def: 4 img
		# if 'dataset_dim' in kwargs:
		# 	self.dataset_dim = kwargs['dataset_dim']
		# else:
		# 	self.dataset_dim = 4

        self.train_split = 5
        self.test_split = 2
        self.val_split = 3

        # se passato in input la dim dei tensori - def: 5x5
        if 'tensor_dim' in kwargs:
            self.tensor_dim = kwargs['tensor_dim']
        else:
            self.tensor_dim = 5

        # set torch seed
        if 'seed' in kwargs:
	        torch.manual_seed(kwargs['seed'])
        else:
            torch.manual_seed(42)

        return

	
    def load_data(self, **kwargs):
        batch_size = 2
        shuffle_train = False
        data_kwargs = {}
        
        # create the tensors
        tensor_list_train = torch.rand(self.train_split, self.tensor_dim, self.tensor_dim)
        tensor_list_test = torch.rand(self.test_split, self.tensor_dim, self.tensor_dim)
        tensor_list_val = torch.rand(self.val_split, self.tensor_dim, self.tensor_dim)
        
        # create the dataset
        train_dataset = TensorDataset(tensor_list_train)
        test_dataset = TensorDataset(tensor_list_test)
        val_dataset = TensorDataset(tensor_list_val)
        
        # create the DataLoader
        self._train_ds = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, **data_kwargs)
        self._val_ds = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **data_kwargs)
        self._test_ds = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **data_kwargs)
        
        self._loaders = {
        'train': self._train_ds,
        'val': self._val_ds,
        'test': self._test_ds
        }
        
        return self._loaders
