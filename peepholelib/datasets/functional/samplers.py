import torch
from torch.utils.data import random_split, Subset, DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
from peepholelib.datasets.datasetWrap import DatasetWrap

class _DSWrap(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {**self.dataset[idx], 'idx': idx}

def random_subsampling(**kwargs):
    ds = kwargs['ds']
    n_samples = kwargs['n_samples']

    assert(isinstance(ds, DatasetWrap))

    if isinstance(n_samples, int):
        n_samples = {k: n_samples for k in ds.__dataset__}
    elif not isinstance(n_samples, dict):
        raise TypeError(f'n_samples must be an int or a dict, got {type(n_samples)}')

    for k, n in n_samples.items():
        ds.__dataset__[k], _ = random_split(ds.__dataset__[k], [n, len(ds.__dataset__[k]) - n])
    return

def balanced_subsampling(**kwargs):
    ds = kwargs['ds']
    n_samples = kwargs['n_samples']
    n_classes = kwargs['n_classes']
    label_key = kwargs.get('label_key', 'label')

    assert(isinstance(ds, DatasetWrap))

    if isinstance(n_samples, int):
        n_samples = {k: n_samples for k in ds.__dataset__}
    elif not isinstance(n_samples, dict):
        raise TypeError(f'n_samples must be an int or a dict, got {type(n_samples)}')

    for k, n in n_samples.items():
        dataset = ds.__dataset__[k]

        n_per_class = max(n//n_classes, 1)
        class_indices = [[] for _ in range(n_classes)]

        for batch in DataLoader(_DSWrap(dataset), batch_size=512, shuffle=True):
            labels = batch[label_key].long()
            idxs = batch['idx']
            for c in range(n_classes):
                needed = n_per_class - len(class_indices[c])
                if needed > 0:
                    class_indices[c] += idxs[labels == c][:needed].tolist()

            if all(len(c) >= n_per_class for c in class_indices):
                break

        ds.__dataset__[k] = Subset(dataset, sum(class_indices, []))
    return
