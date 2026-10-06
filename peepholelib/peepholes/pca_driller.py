from pathlib import Path

import torch
import cupy as cp
from cuml.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from peepholelib.peepholes.drill_base import DrillBase


class PCADriller(DrillBase):
    """Fit a separate PCA for each class at one target layer.

    Uses the usual DrillBase arguments plus reducer to parse corevectors.
    - n_components: defaults to 1; only PC1 is supported for now.
    - plot=True: saves a PC1 explained-variance bar chart after fitting.
    - label_key: defaults to 'label'. No clustering is performed yet.
    - device: passed through to DrillBase, like the other drillers.
      Pass a CUDA device for fitting with cuML; CPU fitting is not supported.

    pcas[class_id] contains the mean, components, explained variance,
    explained variance ratio, and sample count. Components have shape
    (n_components, n_features), in the existing corevector coordinates.
    """

    def __init__(self, *, n_components=1, plot=False, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(self.device)
        if n_components != 1:
            raise NotImplementedError('Only n_components=1 is supported for now.')


        self.n_components = n_components
        self.plot = plot
        self.label_key = kwargs.get('label_key', 'label')
        self.reducer = kwargs['reducer']
        self.cv_parser = self.reducer.parser
        self.pcas = {}
        self._pca_folder = Path(self.path) / self.name
        self._pca_path = self._pca_folder / 'pca.pt'
        self.plot_path = self._pca_folder / 'pc1_explained_variance.png'

    def _parse(self, cvs):
        data = self.cv_parser(cvs=cvs).detach()
        if not torch.isfinite(data).all() or torch.count_nonzero(data) == 0:
            raise ValueError('Corevectors must contain only finite values.')
        return data

    def _check_fitted(self):
        if not self.pcas:
            raise RuntimeError('PCA is not fitted. Run fit() or load() first.')

    def fit(self, **kwargs):
        """Fit on loader='train' using aligned datasets and corevectors.

        Class IDs must be integers in [0, nl_model). Each class needs at least
        two samples and nonzero total variance. Corevectors are centered per
        class, without further standardization or changes to stored data.
        label_key may override the constructor setting.
        Fitting and fitted parameters stay on self.device; DLPack shares
        GPU arrays between PyTorch and CuPy without a CPU round trip.
        """
        if self.device.type != 'cuda':
            raise ValueError('cuML PCA requires a CUDA device, for example device="cuda:0".')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; cuML PCA requires an accessible NVIDIA GPU.')

        loader = kwargs.get('loader', 'train')
        label_key = kwargs.get('label_key', self.label_key)
        datasets = kwargs['datasets']
        corevectors = kwargs['corevectors']
        
        data = self._parse(corevectors._corevds[loader][self.target_module])
        dtype = torch.float64 if data.dtype == torch.float64 else torch.float32
        data = data.to(device=self.device, dtype=dtype)
        labels = datasets._dss[loader][:][label_key].detach().to(data.device).reshape(-1)
        labels = labels.long()

        pcas = {}
        for class_id in range(self.nl_model):
            samples = data[labels == class_id]
            if samples.var(dim=0, unbiased=True).sum() <= 0:
                raise ValueError(f'Class {class_id} has zero variance; PC1 is undefined.')

            with torch.cuda.device(data.device), cp.cuda.Device(data.device.index):

                torch.cuda.current_stream(data.device).synchronize()
                pca = PCA(
                    n_components=self.n_components,
                    svd_solver="jacobi",
                    iterated_power=100,
                    output_type="cupy",
                )                
                pca.fit(cp.from_dlpack(samples), convert_dtype=False)
                torch.cuda.synchronize(data.device)
                pcas[class_id] = {
                    'mean': torch.from_dlpack(pca.mean_).clone(),
                    'components': torch.from_dlpack(pca.components_).clone(),
                    'explained_variance': torch.from_dlpack(pca.explained_variance_).clone(),
                    'explained_variance_ratio': torch.from_dlpack(pca.explained_variance_ratio_).clone(),
                    'n_samples': len(samples),
                }

        self.pcas = pcas
        self.label_key = label_key
        if self.plot:
            self.plot_explained_variance()

    def __call__(self, **kwargs):
        """Project each sample onto every class PCA, without needing labels.

        Returns (n_samples, nl_model, n_components). These are centered PCA
        coordinates, not probabilities or cluster assignments.
        """
        self._check_fitted()
        data = self._parse(kwargs['cvs']).to(
            device=self.device, dtype=self.pcas[0]['mean'].dtype
        )
        return torch.stack([
            (data - self.pcas[j]['mean']) @ self.pcas[j]['components'].T
            for j in range(self.nl_model)
        ], dim=1)

    def plot_explained_variance(self):
        """Save and return the path of the class-wise PC1 variance-ratio plot."""
        self._check_fitted()

        classes = list(range(self.nl_model))
        ratios = [self.pcas[j]['explained_variance_ratio'][0].item() for j in classes]
        fig, ax = plt.subplots(figsize=(max(8, min(20, self.nl_model * 0.3)), 4))
        try:
            ax.bar(classes, ratios)
            ax.set_xlabel('Class')
            ax.set_ylabel('Variance explained by PC1')
            ax.set_title(f'PC1 explained variance by class — {self.target_module}')
            ax.set_ylim(0, 1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.grid(axis='y', alpha=0.3)
            ax.set_axisbelow(True)
            fig.tight_layout()
            self._pca_folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        finally:
            plt.close(fig)
        return self.plot_path

    def save(self, **kwargs):
        """Persist PCA tensors and their layer/class configuration."""
        self._check_fitted()
        self._pca_folder.mkdir(parents=True, exist_ok=True)
        torch.save({
            'target_module': self.target_module,
            'nl_model': self.nl_model,
            'n_features': self.n_features,
            'n_components': self.n_components,
            'label_key': self.label_key,
            'pcas': {
                j: {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in pca.items()}
                for j, pca in self.pcas.items()
            },
        }, self._pca_path)

    def load(self, **kwargs):
        """Load fitted PCAs; return False when no saved file exists."""
        if not self._pca_path.exists():
            return False
        state = torch.load(self._pca_path, weights_only=True, map_location='cpu')
        for key in ('target_module', 'nl_model', 'n_features', 'n_components'):
            if state[key] != getattr(self, key):
                raise ValueError(f'Saved PCA {key}={state[key]!r} does not match {getattr(self, key)!r}.')
        self.pcas = {
            j: {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in pca.items()}
            for j, pca in state['pcas'].items()
        }
        self.label_key = state['label_key']
        return True
