import numpy as np


def compute_cdf_scores(train_scores, train_labels, query_scores, n_classes=100):
    """
    Compute classwise CDF scores for samples 
    against a train reference distribution.

    For each class c, computes the fraction of training samples 
    (of class c) whose score is lower than the query sample's score.

    Args:
        train_scores  (np.ndarray): shape (n_train, n_classes)
        train_labels  (np.ndarray): shape (n_train,)
        query_scores  (np.ndarray): shape (n_query, n_classes)
        n_classes     (int):        number of classes. Defaults to 100.

    Returns:
        np.ndarray: CDF scores, shape (n_query, n_classes), values in [0, 1]
    """
    cdf_out = np.zeros_like(query_scores)
    for c in range(n_classes):
        ref           = train_scores[train_labels == c, c]  # (n_train_c,)
        col           = query_scores[:, c]                  # (n_query,)
        cdf_out[:, c] = (col[:, None] >= ref[None, :]).mean(axis=1)
    return cdf_out


def build_typicality_matrix(peepholes, layers, idx):
    """
    Build a (n_layers, n_classes) typicality matrix for a single sample.

    Args:
        peepholes (PersistentTensorDict): CDF scores tensordict, each key is a layer of shape (n_samples, n_classes)
        layers    (list[str]):            ordered list of layer names
        idx       (int):                  sample index

    Returns:
        np.ndarray: shape (n_layers, n_classes)
    """
    n_classes = peepholes[layers[0]].shape[1]
    mat = np.zeros((len(layers), n_classes))
    for i, layer in enumerate(layers):
        mat[i, :] = peepholes[layer][idx].numpy()
    return mat