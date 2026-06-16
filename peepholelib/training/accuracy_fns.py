# torch stuff
import torch

def img_classification_acc(pred, target, reduction='sum'):
    pred_idx = torch.argmax(pred, dim=1)

    if reduction == 'sum':
        acc = (pred_idx == target).float().sum()
    elif reduction == 'mean':
        acc = (pred_idx == target).float().mean()
    return acc 
