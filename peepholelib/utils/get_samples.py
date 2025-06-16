from torch.nn.functional import softmax
from functools import partial
import torch
from .localization import *
def get_class_names_list(**kwargs):
    """
    Returns a list of class/superclass names for the given dataset.
    """
    ds = kwargs['ds']
    superclass = kwargs.get('superclass', False) #If True, returns superclass names; otherwise, returns class names.

    if ds._classes is None:
        raise RuntimeError("Dataset not loaded. Please run load_data() first.")

    if superclass:
        if ds.dataset != 'CIFAR100':
            raise ValueError("Superclasses are only defined for CIFAR100.")
        return list(ds.get_superclass_mapping().keys())

    # Otherwise, return class names ordered by index
    return [ds._classes[i] for i in sorted(ds._classes)]

def _get_samples_by_class(**kwargs):
    """
    Returns a list of samples for the given class name and split in the dataset.
    Supports class names and superclass names
    """
    ds = kwargs['ds']
    split = kwargs.get('split', 'test') # 'train', 'val', or 'test'
    class_name = kwargs.get('class_name') # Class name or superclass name

    if ds._dss is None:
        raise RuntimeError("Dataset not loaded. Please run load_data() first.")
    if split not in ds._dss:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {list(ds._dss.keys())}")
    split_data = ds._dss[split]

    # Check if class_name is a superclass
    #if ds == 'CIFAR100':
    # superclass_mapping = ds.get_superclass_mapping()
    # if class_name in superclass_mapping:
    #     fine_class_indices = set(superclass_mapping[class_name])
    #     return [i for i, (_, y) in enumerate(split_data) if y in fine_class_indices]

    # Otherwise assume it's a fine class
    class_to_idx = {v: k for k, v in ds._classes.items()}

    if class_name not in class_to_idx:
        raise ValueError(f"Class name '{class_name}' not found in dataset classes.")
    
    target_class_idx = class_to_idx[class_name]
    return [i for i, (_, y) in enumerate(split_data) if y == target_class_idx]
    
def get_filtered_samples(**kwargs):
    """
    Returns a list of sample indices based on:
      - class name (optional)
      - split
      - prediction correctness (optional)
      - confidence range (optional, in %)
      - localization range (optional)

    Notes:
      - If localization_range is provided, you must also provide:
            phs, target_modules
        because localization is computed from peepholes for the chosen layers.
    """
    ds = kwargs["ds"]
    split = kwargs.get("split", "test")
    class_name = kwargs.get("class_name", None)
    correct = kwargs.get("correct", None)
    conf_range = kwargs.get("conf_range", [0, 100])
    pred_fn = kwargs.get("pred_fn", partial(softmax, dim=0))

    # localization filtering (optional)
    localization_range = kwargs.get("localization_range", None)  # e.g. [Lmin, Lmax]
    phs = kwargs.get("phs", None)
    target_modules = kwargs.get("target_modules", None)
    eps = kwargs.get("eps", 1e-12)

    if class_name is None:
        print("No class name provided, returning all samples from split.")
        class_indices = list(range(len(ds._dss[split])))
    else:
        print(f"Filtering samples for class '{class_name}' in split '{split}'")
        class_indices = _get_samples_by_class(ds=ds, split=split, class_name=class_name)

    # fast path
    if correct is None and conf_range == [0, 100] and localization_range is None:
        return class_indices

    # if localization filtering is requested, validate required args
    if localization_range is not None:
        if phs is None or split is None or target_modules is None:
            raise ValueError(
                "To filter by localization_range, you must pass phs, split, and target_modules."
            )
        if not (isinstance(localization_range, (list, tuple)) and len(localization_range) == 2):
            raise ValueError("localization_range must be [min, max].")

    filtered_indices = []
    for idx in class_indices:
        data = ds._dss[split][idx]
        pred = int(data["pred"])
        label = int(data["label"])
        conf = pred_fn(data["output"]).max().item() * 100

        if correct is not None and (pred == label) != correct:
            continue
        if not (conf_range[0] <= conf <= conf_range[1]):
            continue

        # localization filter (optional)
        if localization_range is not None:
            M = torch.stack(
                [phs._phs[split][m]["peepholes"][idx] for m in target_modules],
                dim=0,
            )
            L = float(localization_from_conceptogram(M=M, eps=eps).detach().cpu().item())
            if not (localization_range[0] <= L <= localization_range[1]):
                continue
            else:
                print(f"index {idx} has localization {L}. Adding it to the list.")

        filtered_indices.append(idx)

    print("Filtered_indices:", filtered_indices)
    return filtered_indices