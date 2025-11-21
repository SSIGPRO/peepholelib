# general python stuff
import pandas as pd
from pathlib import Path
from types import NoneType
from math import floor, ceil 
from tqdm import tqdm
import numpy as np

# tensordict
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.datasets.datasetWrap import DatasetWrap
from peepholelib.datasets.sentinel import CustomDS

from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT

# torch stuff
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import mse_loss 

class ConvDataset(Dataset):
    def __init__(self, data, num_sensors=8, seq_len=10):
        self.data = data
        self.num_sensors = num_sensors
        self.seq_len = seq_len

        data.shape[1] == num_sensors * seq_len

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]                  # (80,)
        x = x.view(self.seq_len, self.num_sensors)  # (10, 8)
        x = x.T                             # (8, 10)
        x = x.unsqueeze(0).float()
        #print(f'x.shape{x.shape}') 

        return {"data": x, "label": x}
'''
class CorevectorWrap(DatasetWrap):
    
    def __init__(self, **kwargs):
        self.path = kwargs.get('path')
        return
    
    def __load_data__(self, **kwargs):
        self.__dataset__ = {}
        PersistentTensorDict.from_h5(file_path, mode='r')
'''