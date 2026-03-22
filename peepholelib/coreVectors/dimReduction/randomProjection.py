# General python stuff
from pathlib import Path

# torch stuff
import torch

# Our stuff
from peepholelib.models.model_wrap import get_in_activations
from .dim_reduction_base import DimReductionBase as DRB 
from torch.utils.data import DataLoader

class RandomProjection(DRB):
    """
    Project flattened activations onto a cached random orthonormal basis.
    The projection matrix is sampled from a Gaussian distribution using ``seed`` and stored at ``Path(path) / layer`` for reuse.

    Args:
        path (str or Path): Directory used to store the cached projection.
        layer (str): Name of the layer whose activations are projected.
        model: Wrapped model used to infer the activation dimensionality.
        datasets: Dataset container used to probe one sample activation.
        seed (int): Seed used to generate the random basis.
        rank (int, optional): Projection rank. Defaults to ``300``.
        cv_dim (int, optional): Number of components kept by :meth:`parser`.

    Returns:
        torch.Tensor: Projected activations with shape ``[ns, q]``, where ``ns`` is the batch size and ``q`` is the projection rank.
    """

    def __init__(self, **kwargs):
        """Initialize or load the cached random projection matrix."""
        DRB.__init__(self, **kwargs)
        path = Path(kwargs['path'])
        layer = kwargs['layer']
        model = kwargs['model']
        # datasets = kwargs['datasets'] # testing
        self.seed = kwargs['seed']
        self.q = kwargs.get('rank', 300)
        self.cv_dim = kwargs.get('cv_dim', None)
        sample_in = kwargs.get('sample_in')
        self.verbose = kwargs.get('verbose', False)
        activations_parser = kwargs.get('activations_parser', get_in_activations)
        save_input = kwargs.get('save_input', True)
        save_output = kwargs.get('save_output', False) 

        path.mkdir(parents=True, exist_ok=True)
        self.file_path = path / layer

        _layer = model._target_modules[layer]
        self.device = model.device
        self._random_proj = None
        self.reduct_m = None       

        if self.file_path.exists():
            if self.verbose: print(f'File {self.file_path} exists. Loading from disk.')
            self._random_proj = torch.load(self.file_path)
            self.reduct_m = self._random_proj['reduct_m'].detach().to(self.device)
        else:

            model.set_activations(save_input=save_input, save_output=save_output)

            with torch.no_grad():
                _in = sample_in.to(self.device)
                model(_in)
                _act0 = activations_parser(model._acts)
            
            proj_dim = _act0[layer].flatten(start_dim=1).shape[1]

            model.set_activations(save_input=False, save_output=False)

            generator = torch.Generator(device='cpu')
            generator.manual_seed(self.seed)

            gaussian_m = torch.randn(proj_dim, self.q, generator=generator)
            orthogonal_m, _ = torch.linalg.qr(gaussian_m, mode='reduced')
            orthogonal_m = orthogonal_m.T.contiguous()

            self._random_proj = {
                'reduct_m': orthogonal_m.detach().cpu(),
                'seed': self.seed,
                }
            self.reduct_m = self._random_proj['reduct_m'].detach().to(self.device)

            if self.verbose: print(f'saving {self.file_path}')
            torch.save(self._random_proj, self.file_path)

        return

    def __call__(self, **kwargs):
        """Apply the cached projection matrix to a batch of activations.

        Args:
            act_data (torch.Tensor): Activation batch. All dimensions after the batch dimension are flattened before projection.

        Returns:
            torch.Tensor: Projected activations with shape ``[ns, q]``, where ``ns`` is the batch size and ``q`` is the projection rank.
        """

        act_data = kwargs['act_data'] 

        _acts = act_data.flatten(start_dim=1)
        cvs = (self.reduct_m@_acts.T).T

        return cvs

    def parser(self, **kwargs):
        """
        
        Trim projected vectors and optionally return their labels.

        Args:
            cvs (torch.Tensor): Batch of projected activations with shape ``[ns, q]``.
            dss (TensorDict, optional): Dataset batch aligned with ``cvs``. When provided, labels are returned alongside the trimmed corevectors.
            label_key (str, optional): Key used to read labels from ``dss``. Defaults to ``"label"``.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Trimmed corevectors with shape ``[ns, cv_dim]``. If ``dss`` is provided, returns ``(trimmed_corevectors, labels)``.
        """

        cvs = kwargs['cvs']
        dss = kwargs.get('dss', None)
        label_key = kwargs.get('label_key', 'label') 

        # trim corevectors on the last dimension
        tcvs = cvs[...,0:self.cv_dim]

        ret = tcvs if dss == None else (tcvs, dss[label_key])
        return ret