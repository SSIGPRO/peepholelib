import torch

def random_projection(**kwargs):
    '''
    Random projection for activations of a layer.

    Args:
        - act_data (torch.tensor): batched input activations
        - layer (torch.nn.Conv2d or torch.nn.Linear): layer 
        - device (torch.device): device to perform computations. If None, inferred from act_data.
        - proj (torch.tensor, optional): precomputed projection matrix of shape (in_dim, proj_dim).
        - proj_dim (int): target dimensionality of the random projection.

     Returns:
        - cvs (torch.tensor): batched projected activations, shape (B, proj_dim)
    '''
    act_data = kwargs.get('act_data', None)
    layer = kwargs.get('layer', None)
    device = kwargs.get('device', None)
    proj = kwargs.get('proj', None)
    proj_dim = kwargs.get('proj_dim', 100)

    if proj is None:
        print("Careful.. random projection matrix will be different each time unless you provide one.")
    
    if act_data is None:
        raise ValueError("act_data must be provided to random_projection().")

    act_data = act_data.to(device)
    ns = act_data.shape[0]
    acts_flat = act_data.flatten(start_dim=1)

    if getattr(layer, "bias", None) is not None:
        ones  = torch.ones(ns, 1, device=device, dtype=acts_flat.dtype)
        _acts = torch.hstack((acts_flat, ones))                # (ns, Din+1)
    else:
        _acts = acts_flat                                      # (ns, Din)

    in_dim = _acts.shape[1]

    proj = proj.to(device=device, dtype=_acts.dtype)
    if proj.shape[0] != in_dim:
        raise ValueError(
            f"Projection matrix has in_dim={proj.shape[0]} but activations have in_dim={in_dim}. Layer: {layer}."
        )

    cvs = _acts @ proj
    return cvs