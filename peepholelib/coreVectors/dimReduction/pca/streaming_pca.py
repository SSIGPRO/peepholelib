# General python stuff
import hashlib
from math import ceil
from pathlib import Path

# torch stuff
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Our stuff
from peepholelib.models.model_wrap import get_in_activations
from ..dim_reduction_base import DimReductionBase as DRB


class StreamingPCA(DRB):
    def __init__(self, **kwargs):
        DRB.__init__(self, **kwargs)
        path = Path(kwargs["path"])
        layer = kwargs["layer"]
        model = kwargs["model"]
        dataset = kwargs["dataset"]
        ds_key = kwargs["ds_key"]
        self.cv_dim = kwargs["cv_dim"]
        rank = kwargs.get("rank", 300)
        self.input_key = kwargs.get("input_key", "image")
        self.batch_size = kwargs.get("batch_size", 64)
        self.n_threads = kwargs.get("n_threads", 1)
        self.activations_parser = kwargs.get("activations_parser", get_in_activations)
        verbose = kwargs.get("verbose", False)

        path.mkdir(parents=True, exist_ok=True)

        _layer = model._target_modules[layer]
        self.layer = layer
        self.layer_module = _layer
        self.device = model.device

        file_path = path / f"{layer}.{ds_key}.streaming_pca.pt"

        if file_path.exists():
            if verbose:
                print(f"File {file_path} exists. Loading from disk.")
            self._pca = torch.load(file_path, weights_only=False)
        else:
            self._pca = self._fit_from_dataset(
                dataset=dataset,
                ds_key=ds_key,
                model=model,
                rank=rank,
                verbose=verbose,
            )

            if verbose:
                print(f"saving {file_path}")
            torch.save(self._pca, file_path)

        self.rank = self._pca["components"].shape[0]
        if self.cv_dim is None:
            self.cv_dim = self.rank

        self.mean = self._pca["mean"].detach().to(self.device)
        self.reduct_m = self._pca["components"].detach().to(self.device)

        return

    def _fit_from_dataset(self, **kwargs):
        dataset = kwargs["dataset"]
        ds_key = kwargs["ds_key"]
        model = kwargs["model"]
        rank = kwargs["rank"]
        verbose = kwargs.get("verbose", False)

        if dataset._dss is None or ds_key not in dataset._dss:
            raise RuntimeError(
                f"Dataset split '{ds_key}' is not available in ParsedDataset._dss. "
                "Load the dataset and keep it open within its context manager before fitting the PCA."
            )

        ds = dataset._dss[ds_key]
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_threads,
            collate_fn=lambda x: x
        )

        n_seen = 0
        mean = None
        singular_values = None  # [k]
        components = None       # [k, D]

        model.set_activations(save_input=True, save_output=False)
        try:
            with torch.no_grad():
                for batch in tqdm(dl, desc=f"Fitting PCA for layer {self.layer}"):

                    model(batch[self.input_key].to(model.device))
                    acts = self.activations_parser(model._acts)[self.layer]
                    acts = acts.flatten(start_dim=1).detach().to(dtype=torch.float64, device="cpu")

                    n_batch = acts.shape[0]
                    if n_batch == 0:
                        continue

                    batch_mean = acts.mean(dim=0)

                    if mean is None:
                        # First batch: initialise with a thin SVD of the centred batch.
                        mean = batch_mean
                        n_seen = n_batch
                        _, S, Vt = torch.linalg.svd(acts - batch_mean, full_matrices=False)
                        k = min(rank, S.shape[0])
                        singular_values = S[:k]
                        components = Vt[:k]
                    else:
                        n_total = n_seen + n_batch

                        # Correction vector that accounts for the shift in the running mean.
                        # Equivalent to an extra "virtual" data point that encodes how much
                        # the old components need to move to be centred at the new mean.
                        mean_correction = torch.sqrt(
                            torch.tensor(n_seen * n_batch / n_total, dtype=torch.float64)
                        ) * (mean - batch_mean)

                        mean = (n_seen * mean + n_batch * batch_mean) / n_total

                        # Augmented matrix: old basis (scaled by singular values) +
                        # new centred batch + mean correction row.
                        # Shape: (k + n_batch + 1, D) — O(rank * D), not O(D^2).
                        X_aug = torch.cat([
                            singular_values[:, None] * components,  # [k, D]
                            acts - batch_mean,                       # [n_batch, D]
                            mean_correction.unsqueeze(0),            # [1, D]
                        ], dim=0)

                        _, S_new, Vt_new = torch.linalg.svd(X_aug, full_matrices=False)
                        k = min(rank, S_new.shape[0])
                        singular_values = S_new[:k]
                        components = Vt_new[:k]

                        n_seen = n_total

                    if verbose:
                        print(f"processed {n_seen} samples for {self.layer}")
        finally:
            model.set_activations(save_input=False, save_output=False)

        if mean is None or n_seen < 2:
            raise RuntimeError(
                f"Need at least 2 samples to fit streaming PCA on split '{ds_key}'. Got {n_seen}."
            )

        explained_variance = singular_values ** 2 / (n_seen - 1)

        return {
            "mean": mean.to(dtype=torch.float32),
            "components": components.to(dtype=torch.float32),
            "explained_variance": explained_variance.clamp_min(0).to(dtype=torch.float32),
            "n_samples": n_seen,
        }

    def __call__(self, **kwargs):
        """
        Applies the PCA projection to activations from the selected layer.
        The output has shape `[ns, rank]`, where `ns` is the number of samples in the batch.

        Args:
        - act_data (torch.tensor): batched input activations

        Returns:
        - cvs (torch.tensor): batched projected activations
        """

        act_data = kwargs["act_data"]
        acts_flat = act_data.flatten(start_dim=1)
        centered = acts_flat - self.mean.to(acts_flat.device)
        cvs = centered @ self.reduct_m.to(acts_flat.device).T

        return cvs

    def parser(self, **kwargs):
        """
        Trims corevectors obtained with `LinearStreamingPCA`.
        Input shape is `[ns, rank]`, where `ns` is the number of samples in the batch.
        Output shape is `[ns, self.cv_dim]`.

        Args:
            cvs (TensorDict): Batch from TensorDict for corevectors inside `peepholelib.CoreVectors` class.
            dss (TensorDict): Batch from TensorDict for dataset inside `peepholelib.CoreVectors` class
            label_key (str): key to get labels from

        Returns:
            tcvs (torch.tensor): Trimmed corevectors and corresponding labels
            labels (torch.tensor): Labels from dataset for the samples. Only returned if `dss` is given
        """

        cvs = kwargs["cvs"]
        dss = kwargs.get("dss", None)
        label_key = kwargs.get("label_key", "label")

        tcvs = cvs[..., 0:self.cv_dim]

        ret = tcvs if dss is None else (tcvs, dss[label_key])
        return ret
