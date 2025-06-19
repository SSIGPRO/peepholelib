def trim_corevectors(**kwargs):
    """
    Trims corevectors obtained with `coreVectors.dimReduction.svds.conv2d_svd_projection(), linear_svd_projection_ViT(), conv2d_toeplitz_svd_projection(channel_wise=False)`.
    Input shape is `[ns, q]`, where `ns` is the number of samples in the batch, `q` the SVD rank.
    Output shape is `[ns, cv_dim]`, the concatenation of the trimmed corevectors of all output channels.
                                                                                                                    
    Args:
        cvs (TensorDict): Batch from TensorDict for corevectors inside `peepholelib.CoreVectors` class.
        dss (TensorDict): Batch from TensorDict for dataset inside `peepholelib.CoreVectors` class
        module (str): target module key.
        label_key (str): key to get labels from
        cv_dim (int): desired dimension of corevector
                                                                                                                    
    Returns:
        tcvs (torch.tensor): Trimmed corevectors and correspective labels
        labels (torch.tensor): Labels from datasate for the samples. Only returned if `dss` is given
    """

    cvs = kwargs['cvs']
    dss = kwargs['dss'] if 'dss' in kwargs else None
    module = kwargs['module']
    label_key = kwargs['label_key'] if 'label_key' in kwargs else 'label' 
    cv_dim = kwargs['cv_dim']

    # trim corevectors on the last dimension
    tcvs = cvs[module][...,0:cv_dim]

    if dss == None:
        return tcvs
    else:
        labels =  dss[label_key]
        return tcvs, labels  

def trim_channelwise_corevectors(**kwargs):
    """
    Trims multi channel corevectors obtained with `coreVectors.dimReduction.svds.conv2d_toeplitz_svd_projection()`.
    Input shape is `[ns, cout, q]`, where `ns` is the number of samples in the batch, `cout` the number of output channels of the layer, and `q` the SVD rank.
    Output shape is `[ns, cout*cv_dim]`, the concatenation of the trimmed corevectors of all output channels.

    Args:
        cvs (TensorDict): Batch from TensorDict for corevectors inside `peepholelib.CoreVectors` class.
        dss (TensorDict): Batch from TensorDict for dataset inside `peepholelib.CoreVectors` class
        module (str): target module key.
        label_key (str): key to get labels from
        cv_dim (int): desired dimension of corevector

    Returns:
        tcvs (torch.tensor): Trimmed corevectors and correspective labels
        labels (torch.tensor): Labels from datasate for the samples. Only returned if `dss` is given
    """

    cvs = kwargs['cvs']
    dss = kwargs['dss'] if 'dss' in kwargs else None
    module = kwargs['module']
    label_key = kwargs['label_key'] if 'label_key' in kwargs else 'label' 
    cv_dim = kwargs['cv_dim']
    cols = kwargs['cols'] if 'cols' in kwargs else None
    
    _cv = cvs[module]
    _ns = _cv.shape[0] # n samples
    _nc = _cv.shape[1] # n channels
    _r  = _cv.shape[2] # rank
    
    # trim and reshape cvs
    if cols is None:
        _tcv = _cv[:,:,0:cv_dim]
        trcv = _tcv.reshape(_ns, _nc*cv_dim)
    else:
        _tcv = _cv[:,cols,0:cv_dim]
        trcv = _tcv.reshape(_ns, len(cols)*cv_dim)

    if dss == None:
        return trcv 
    else:
        labels =  dss[label_key]
        return trcv, labels 

def trim_kernel_corevectors(**kwargs):
    """
    Trims multi kernel corevectors obtained with `coreVectors.dimReduction.svds.conv2d_kernel_svd_projection()`.
    Input shape is `[ns, ow*oh, q]`, where `ns` is the number of samples in the batch, `ow, oh` the width and heigth of the layer's output, and `q` the SVD rank.
    Output shape is `[ns, ow*oh*cv_dim]`, the concatenation of the trimmed corevectors of all output channels.

    Args:
        cvs (TensorDict): Batch from TensorDict for corevectors inside `peepholelib.CoreVectors` class.
        dss (TensorDict): Batch from TensorDict for dataset inside `peepholelib.CoreVectors` class
        module (str): target module key.
        label_key (str): key to get labels from
        cv_dim (int): desired dimension of corevector

    Returns:
        tcvs (torch.tensor): Trimmed corevectors and correspective labels
        labels (torch.tensor): Labels from datasate for the samples. Only returned if `dss` is given
    """

    cvs = kwargs['cvs']
    dss = kwargs['dss'] if 'dss' in kwargs else None
    module = kwargs['module']
    label_key = kwargs['label_key'] if 'label_key' in kwargs else 'label' 
    cv_dim = kwargs['cv_dim']
    
    _cv = cvs[module]
    _ns = _cv.shape[0] # n samples
    _ks = _cv.shape[1] # n kernels 
    _r  = _cv.shape[2] # rank
    
    # trim and reshape cvs
    _tcv = _cv[...,0:cv_dim]
    trcv = _tcv.reshape(_ns, _ks*cv_dim)
    
    if dss == None:
        return trcv 
    else:
        labels =  dss[label_key]
        return trcv, labels 

def get_images(**kwargs):
    """
    Get only images

    Args:
        act (PersistentTensorDict): TensorDict for the activations inside `peepholelib.CoreVectors` class

    Returns:
        image (torch.Tensor): images saved in the activations.  
    """
    img = kwargs['dss']['image']
    return img 
