# torch stuff
import torch

def compute_top_k_accuracy(peepholes, targets, k):
    """
    Compute the topk accuracy given classification probabilities (shape 'n_samples*n_labels').
    
    Args:
    - peepholes (torch.tensor): probabilities of each sample to belong to each label
    - targets (torch.tensor): labels for all samples
    - k (int): top-k to compute
    """
    topk = torch.topk(peepholes, k=k, dim=1).indices
    targets = targets.unsqueeze(1)
    correct = (topk == targets).any(dim=1).float()
    return correct.mean().item()
