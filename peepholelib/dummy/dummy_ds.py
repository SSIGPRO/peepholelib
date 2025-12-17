import torch
from torch.utils.data import Dataset
from peepholelib.datasets.dataset_base import DatasetBase

class Silly(Dataset):
    def __init__(self, n, ds):
        self.n = n
        self.ds = ds

        self.data = torch.rand((n,)+self.ds)
        self.labels = torch.randint(0, 10, (self.n,))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DummyDS(DatasetBase):
    def __init__(self, **kwargs):
        self.n = kwargs['n_samples']
        self.ds = kwargs['data_size']
        return

    def load_data(self, **kwargs):
        self._dss = {'d': Silly(self.n, self.ds)}
        self._classes = torch.linspace(0, 9, 10)  
        return

