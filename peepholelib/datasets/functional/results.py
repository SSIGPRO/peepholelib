import torch

def results_one_hot_encoding(predicted_labels, true_labels):
    return predicted_labels == true_labels

def results_multiple_hot_encoding(predicted_labels, true_labels):
    print(predicted_labels.shape, true_labels.shape)
    return true_labels[torch.arange(predicted_labels.shape[0]), predicted_labels] == 1