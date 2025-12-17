import torch

def binary_classification(output):
    return torch.sigmoid(output).squeeze() > 0.5

def multilabel_classification(output):
    return torch.argmax(output,axis=1)
