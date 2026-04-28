# General python stuff
from pathlib import Path

# torch stuff
import torch

# Our stuff
from ..dim_reduction_base import DimReductionBase as DRB 

class ViTLinearSVD(DRB):

    _REDUCTIONS = {
        'first': lambda act: act[:, 0, :],
        'mean':  lambda act: act.mean(dim=1),
    }

    def __init__(self, **kwargs):
        DRB.__init__(self, **kwargs)
        path = Path(kwargs['path'])
        model = kwargs['model']
        layer = kwargs['layer']
        q = kwargs.get('rank',300)
        self.cv_dim = kwargs.get('cv_dim', None)
        token_reduction = kwargs.get('token_reduction', 'first')
        if token_reduction not in self._REDUCTIONS:
            raise RuntimeError(f"Unknown token_reduction '{token_reduction}'. Supported: {list(self._REDUCTIONS)}")
        self.red_fn = self._REDUCTIONS[token_reduction]

        verbose = kwargs.get('verbose', False)
                                                      
        # create folder
        path.mkdir(parents=True, exist_ok=True)
        file_path = path/layer

        # get ref for the layer
        _layer = model._target_modules[layer]
        device = model.device

        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._svd = torch.load(file_path)
        else: 
            # computation
            W = _layer.weight
            use_bias = _layer.bias is not None
            if use_bias:
                W = torch.hstack((W, _layer.bias.reshape(-1, 1)))
            W = W.to(device)
            U, s, Vh = torch.svd_lowrank(W, q)
            U, s, Vh = U.detach().cpu(), s.detach().cpu(), Vh.detach().cpu()
            self._svd = {
                    'U': U,
                    's': s,
                    'Vh': Vh.T,
                    'use_bias': use_bias
                    }

            if verbose: print(f'saving {file_path}')
            torch.save(self._svd, file_path)

        # save variables used in the projection a.k.a. "__call__()"
        self.reduct_m = self._svd['Vh'].detach().to(device)
        in_features = _layer.weight.shape[1]
        in_dim = self.reduct_m.shape[1]
        if in_dim == in_features + 1:
            self.use_bias = True
        elif in_dim == in_features:
            self.use_bias = False
        else:
            raise RuntimeError(f"Loaded SVD input dimension ({in_dim}) does not match layer input dimension ({in_features}) for layer {layer}.")
        
        return
        
            
    def __call__(self, **kwargs):
        '''
        Applies the SVD projection to `torch.Linear` activations. The output has shape `[ns, q]`, where `ns` is the number of samples in the batch, and `q` the SVD rank.
        For tokenized inputs `[ns, nt, c]`, `token_reduction` controls how tokens are reduced:
        - 'first': first token (ViT class token style)
        - 'mean': mean over tokens (useful for models without class token, e.g. Swin)
        For Swin qkv fallbacks, 4D activations `[ns, h, w, c]` are also supported and converted to `[ns, h*w, c]` before token reduction.

        Args:
        - act_data (torch.tensor): batched input activations

        Returns:
        - cvs (torch.tensor) = batched projected activations
        '''
        act_data = kwargs['act_data']
        n_act = act_data.shape[0]

        if act_data.ndim == 4:
            act_data = act_data.flatten(start_dim=1, end_dim=2)
        if act_data.ndim == 3:
            act_data = self.red_fn(act_data)
        elif act_data.ndim != 2:
            raise RuntimeError(f"Expected 2D/3D/4D activations, got shape {tuple(act_data.shape)}.")

        acts_flat = act_data.flatten(start_dim=1)
        if self.use_bias:
            _acts = torch.hstack((acts_flat, torch.ones(n_act, 1, device=acts_flat.device)))
        else:
            _acts = acts_flat

        return (self.reduct_m @ _acts.T).T


    def parser(self, **kwargs):
        """
        Trims corevectors obtained with `coreVectors.dimReduction.svds.vit_linear_svd.ViTLinearSVD.
        Input shape is `[ns, q]`, where `ns` is the number of samples in the batch, `q` the SVD rank.
        Output shape is `[ns, self.cv_dim]`, trimmed corevectors
                                                                                                            
        Args:
            cvs (TensorDict): Batch from TensorDict for corevectors inside `peepholelib.CoreVectors` class.
            dss (TensorDict): Batch from TensorDict for dataset inside `peepholelib.CoreVectors` class
            label_key (str): key to get labels from
                                                                                                            
        Returns:
            tcvs (torch.tensor): Trimmed corevectors and correspective labels
            labels (torch.tensor): Labels from datasate for the samples. Only returned if `dss` is given
        """

        cvs = kwargs['cvs']
        dss = kwargs.get('dss', None)
        label_key = kwargs.get('label_key', 'label') 

        # trim corevectors on the last dimension
        tcvs = cvs[...,0:self.cv_dim]

        ret = tcvs if dss == None else (tcvs, dss[label_key])
        return ret 