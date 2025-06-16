from pathlib import Path
import math

import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

def get_random_projs(self, **kwargs):
    """
    Compute (or load) random projection matrices for the given target modules.

    Each layer mk gets a projection matrix of shape [in_dim, proj_dim], where:
      - in_dim is the flattened input activation dimension for that layer
      - proj_dim is a user-specified target dimension.

    Args:
    - path (Path or str): directory to save/load the projections.
    - name (str): file name for the memmapped TensorDict.
    - target_modules (list[str]): layer names (keys in self._target_modules).
    - proj_dim (int): target dimensionality of the random projection.
    - sample_in (torch.Tensor): a single sample input (CHW) used for a dry run
                                to infer activation shapes.
    - verbose (bool): print progress messages.

    Returns:
    - self._rand_projs (TensorDict):
        { layer_name: { 'proj': <MMT-wrapped projection matrix> } }
    """

    path           = Path(kwargs.get('path'))
    name           = kwargs.get('name')
    target_modules = kwargs.get('target_modules')
    proj_dim       = kwargs.get('proj_dim')
    sample_in      = kwargs.get('sample_in')
    verbose        = kwargs.get('verbose', False)

    if proj_dim is None:
        raise ValueError("proj_dim must be provided to get_random_projs().")

    # create folder
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / name
    if file_path.exists():
        if verbose:
            print(f'File {file_path} exists. Loading random projections from disk.')
        _projs = TensorDict.load_memmap(file_path)
    else:
        _projs = TensorDict()

    # figure out which modules still need projections
    _modules_to_compute = []
    for mk in target_modules:
        if mk in _projs.keys():
            continue
        _modules_to_compute.append(mk)
    if verbose:
        print('modules to compute random projections: ', _modules_to_compute)

    if len(_modules_to_compute) == 0:
        # nothing new to do
        self._rand_projs = _projs
        return self._rand_projs

    self.set_activations(save_input=True, save_output=False)

    # Dry run to populate self._acts['in_activations'] with shapes
    with torch.no_grad():
        _in = sample_in.reshape((1,) + sample_in.shape).to(self.device)
        self(_in)

    for mk in _modules_to_compute:
        if verbose:
            print(f'\n ---- Getting random projection for {mk}\n')
        module = self._target_modules[mk]

        in_act = self._acts['in_activations'][mk]   # shape [1, ...]
        in_flat = in_act.flatten(start_dim=1)       # shape [1, in_dim_no_bias]
        in_dim = in_flat.shape[1]

        if getattr(module, 'bias', None) is not None:
            in_dim = in_dim + 1

        if verbose:
            print(f'  inferred in_dim = {in_dim}, proj_dim = {proj_dim}')

        # Create random projection matrix [in_dim, proj_dim]
        proj = torch.randn(
            in_dim,
            proj_dim,
            device=self.device,
            dtype=in_act.dtype
        ) / math.sqrt(in_dim)

        proj = proj.detach().cpu()
        _projs[mk] = TensorDict({
            'proj': MMT(proj)
        })

    self.set_activations(save_input=False, save_output=False)

    if verbose:
        print(f'saving {file_path}')
    if len(_modules_to_compute) != 0:
        _projs.memmap(file_path)

    self._rand_projs = _projs
    return self._rand_projs