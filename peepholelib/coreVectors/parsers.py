import torch

def from_dataset(batch):
    images, labels = zip(*batch)
    return {'image': images, 'label': torch.tensor(labels)}

