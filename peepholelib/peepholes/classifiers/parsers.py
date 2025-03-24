def trim_corevectors(**kwargs):
    """
    Trims peephole data from a give layer.

    Args:
      tensor_dict (TensorDict): TensorDict from our CoreVectors class.
      layer (str): Layer key.

    Returns:
        nothing 
    """
    cvs = kwargs['cvs']
    act = kwargs['act'] if 'act' in kwargs else None
    layer = kwargs['layer']
    label_key = kwargs['label_key'] if 'label_key' in kwargs else 'label' 
    cv_dim = kwargs['cv_dim']

    if act == None:
        return cvs[layer][:,0:cv_dim]
    else:
        return cvs[layer][:,0:cv_dim], act[label_key]
