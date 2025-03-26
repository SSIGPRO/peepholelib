import torch

def from_dataset(batch, key_list):
    images, labels = zip(*batch)
    return {'image': images, 'label': torch.tensor(labels)}
