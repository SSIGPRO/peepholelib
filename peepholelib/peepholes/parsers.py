def trim_corevectors(**kwargs):
    """
    Trims peephole data from a give module.

    Args:
        cvs (PersistentTensorDict): TensorDict for corevectors inside `peepholelib.CoreVectors` class.
        act (PersistentTensorDict): TensorDict for the activations inside `peepholelib.CoreVectors` class
        module (str): target module key.
        label_key (str): key to get labels from
    Returns:
        Trimmed activations and correspective labels 

    """
    cvs = kwargs['cvs']
    act = kwargs['act'] if 'act' in kwargs else None
    module = kwargs['module']
    label_key = kwargs['label_key'] if 'label_key' in kwargs else 'label' 
    cv_dim = kwargs['cv_dim']

    if act == None:
        return cvs[module][:,0:cv_dim]
    else:
        return cvs[module][:,0:cv_dim], act[label_key]  

def get_images(**kwargs):
    """
    Get only images

    Args:
        act (PersistentTensorDict): TensorDict for the activations inside `peepholelib.CoreVectors` class

    Returns:
        image (torch.Tensor): images saved in the activations.  
    """
    img = kwargs['act']['image']
    return img 
