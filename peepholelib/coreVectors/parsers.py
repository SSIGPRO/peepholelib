import torch

def from_dataset(batch, key_list):

    images, labels = zip(*batch)
    
    images, labels = torch.stack(images), torch.tensor(labels)
    
    
    return {key_list[0]: images, key_list[1]: labels}

