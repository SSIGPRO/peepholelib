def trim_corevectors(**kwargs):
    """
    Trims corevectors from a give module.

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

def trim_channelwise_corevectors(**kwargs):
    """
    Trims multi channel corevectors from a give module. E.G., Conv2D corevectors with `channel_wise=True`.
    Each row in these corevectors is one channel, and colums are the rank, so we trim alongside collumns.
    Output is concatenated.

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
    cols = kwargs['cols'] if 'cols' in kwargs else None
    
    _cv = cvs[module]
    _ns = _cv.shape[0] # n samples
    _nc = _cv.shape[1] # n channels
    _r  = _cv.shape[2] # rank
    
    if cols is None:
        _tcv = _cv[:,:,0:cv_dim] # timmed cv
        _trcv = _tcv.reshape(_ns, _nc*cv_dim) # trimmed and reshaped cv
    else:
        _tcv = _cv[:,cols,0:cv_dim] # timmed cv
        _trcv = _tcv.reshape(_ns, len(cols)*cv_dim) # trimmed and reshaped cv

    if act == None:
        return _trcv 
    else:
        return _trcv, act[label_key]  

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
