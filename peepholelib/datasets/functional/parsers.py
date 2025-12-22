import torch

def from_dataset(batch, keylist=['image', 'label']):

    features = list(zip(*batch))

    parsed_batch = {}

    for f, key in zip(features, keylist):
        print(f)
        parsed_batch[key] = torch.stack(f)

    return parsed_batch