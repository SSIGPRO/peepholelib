# General python stuff
from pathlib import Path

# torch stuff
import torch

# Our stuff
from ..dim_reduction_base import DimReductionBase as DRB 
#from .clustering_optimization import optimize_clustering_projection

class LinearSVD(DRB):
    def __init__(self, **kwargs):
        DRB.__init__(self, **kwargs)
        path = Path(kwargs['path'])
        layer = kwargs['layer']
        model = kwargs['model']
        q = kwargs.get('rank', 300)
        self.cv_dim = kwargs['cv_dim']
        self.layer_name = layer
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
            W = torch.hstack((_layer.weight, _layer.bias.reshape(-1,1))).to(device)
            U, s, Vh = torch.svd_lowrank(W, q)
            U, s, Vh = U.detach().cpu(), s.detach().cpu(), Vh.detach().cpu()
            self._svd = {
                    'U': U,
                    's': s,
                    'Vh': Vh.T
                    }

            if verbose: print(f'saving {file_path}')
            torch.save(self._svd, file_path)
        
        # save variables used in the projection a.k.a. "__call__()"
        self.reduct_m = self._svd['Vh'].detach().to(device)

        return

    def _prepare_projection_input(self, act_data):
        n_act = act_data.shape[0]
        acts_flat = act_data.flatten(start_dim=1)
        ones = torch.ones(n_act, 1, device=acts_flat.device)
        return torch.hstack((acts_flat, ones))
            
    def __call__(self, **kwargs):
        '''
        Applies the SVD projection to `torch.Linear` activations. The output has shape `[ns, q]`, where `ns` is the number of samples in the batch, and `q` the SVD rank.

        Args:
        - act_data (torch.tensor): batched input activations

        Returns:
        - cvs (torch.tensor) = batched projected activations
        '''

        act_data = kwargs['act_data']
        _acts = self._prepare_projection_input(act_data)
        cvs = (self.reduct_m@_acts.T).T

        return cvs

    def optimize_clustering(self, **kwargs):
        """
        Optimize the current V^T projection for clustering.

        Args:
        - act_data (torch.tensor): Batched activations.
        - inplace (bool): If True, update self.reduct_m with the optimized
          projection. Defaults to False.
        - datasets (ParsedDataset, optional): Parsed dataset container used to
          read class labels from `datasets._dss[loader][label_key]`. When
          provided, optimization metrics also include class and cluster
          coverage computed from the empirical posterior mapping.
        - loader (str): Dataset split key used with `datasets`. Defaults to
          `'train'`.
        - label_key (str): Label field inside the parsed dataset. Defaults to
          `'label'`.
        - remaining kwargs: forwarded to optimize_clustering_projection().
        """
        act_data = kwargs.pop('act_data')
        inplace = kwargs.pop('inplace', False)
        h_data = self._prepare_projection_input(act_data)
        results = optimize_clustering_projection(
                h_data=h_data,
                reduct_m=self.reduct_m,
                cv_dim=self.cv_dim,
                layer_name=self.layer_name,
                **kwargs
                )
        if inplace:
            self.reduct_m = results['optimized_reduct_m'].detach().to(self.reduct_m.device)
            self._svd['Vh'] = self.reduct_m.detach().cpu()
            self.cv_dim = int(results.get('optimized_cv_dim', self.cv_dim))

        return results

    def parser(self, **kwargs):
        """
        Trims corevectors obtained with `coreVectors.dimReduction.svds.linear_svd.LinearSVD.
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
