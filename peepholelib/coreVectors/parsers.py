import torch

def from_dataset(batch):
    images, labels = zip(*batch)
    images, labels = torch.stack(images), torch.tensor(labels)
    return {'image': images, 'label': labels}

