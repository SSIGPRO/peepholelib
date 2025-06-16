import pickle
from torch.nn.functional import softmax
from functools import partial

from peepholelib.utils.mappings import coarse_to_fine_cifar100 as coarse_to_fine

def get_class_names_list(ds, superclass = False):
    if ds._classes is None:
        raise RuntimeError("Dataset not loaded. Please run load_data() first.")

    if not superclass:
        return [ds._classes[i] for i in sorted(ds._classes)]

    if ds.dataset != 'CIFAR100':
        raise ValueError("Superclasses are only defined for CIFAR100.")

    meta_path = '/srv/newpenny/dataset/CIFAR100/cifar-100-python/meta'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='latin1')

    return meta['coarse_label_names']

def get_samples(ds, split: str, class_name: str):
    """
    Args:
        ds (DatasetBase): e.g., Cifar.
        split (str): 'train', 'val', or 'test'.
        class_name (str): Class name or superclass name.

    Returns:
        List[int]: List of samples' indices of class_name and split.
    """
    if ds._dss is None:
        raise RuntimeError("Dataset not loaded. Please run load_data() first.")
    if split not in ds._dss:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {list(ds._dss.keys())}")

    split_data = ds._dss[split]

    if ds.dataset == 'CIFAR100':
        # Load CIFAR-100 meta
        meta_path = '/srv/newpenny/dataset/CIFAR100/cifar-100-python/meta'
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='latin1')

        fine_to_idx = {v: k for k, v in ds._classes.items()}

        # if superclass
        if class_name in coarse_to_fine: 
            fine_classes = coarse_to_fine[class_name]
            target_class_indices = {fine_to_idx[c] for c in fine_classes}
            return [i for i, (_, y) in enumerate(split_data) if y in target_class_indices]
    
    # if not superclass
    class_to_idx = {v: k for k, v in ds._classes.items()}
    if class_name not in class_to_idx:
        raise ValueError(f"Class name '{class_name}' not found in dataset classes.")
    
    target_class_idx = class_to_idx[class_name]
    return [i for i, (_, y) in enumerate(split_data) if y == target_class_idx]

def get_filtered_samples(**kwargs):
    """
    Args:
        ds: Dataset
        cvs: Corevectors (only required when filtering by 'correct' or 'conf_range')
        split (str): 'train', 'val' or 'test' (deault: 'test').
        class_name (str): Name of the class or superclass.
        correct (bool): correct or incorrect predicted samples.
        conf_range (list): Confidence range [min, max] (in percent, 0–100 scale).
        pred_fn (callable): Function to convert model output to probabilities (default: softmax).

    Returns:
        List[int]: List of sample indices based on class name, split, prediction, and confidence range
    """
    ds = kwargs['ds']
    split = kwargs.get('split', 'test')
    class_name = kwargs.get('class_name', None)
    correct = kwargs.get('correct', None)
    conf_range = kwargs.get('conf_range', [0, 100]) 
    pred_fn = kwargs.get('pred_fn', partial(softmax, dim=0))

    if class_name is None:
        class_indices = list(range(len(ds._dss[split]))) # all samples from split
    else:
        class_indices = get_samples(ds, split, class_name) 
    
    if correct is None and conf_range == [0, 100]:
        return class_indices

    cvs = kwargs.get('corevectors')
    if cvs is None:
        raise ValueError("Missing 'corevectors'. Required when filtering by 'correct' or 'conf_range'.")

    filtered_indices = []
    for idx in class_indices:
        data = cvs._dss[split][idx]
        pred = int(data['pred'])
        label = int(data['label'])
        conf = pred_fn(data['output']).max().item() * 100

        if correct is not None and (pred == label) != correct:
            continue
        if not (conf_range[0] <= conf <= conf_range[1]):
            continue

        filtered_indices.append(idx)

    return filtered_indices