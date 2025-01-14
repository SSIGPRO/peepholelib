from torch.utils.data import random_split, DataLoader

def trim_dataloaders(dls, perc):
    out_dls = {}
    for k in dls:
        a, _ = random_split(dls[k].dataset, [perc, 1.0-perc])
        out_dls[k] = DataLoader(a, batch_size = dls[k].batch_size, collate_fn=dls[k].collate_fn)
    return out_dls
