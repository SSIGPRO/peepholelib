from pathlib import Path

import torch

from peepholelib.models.model_wrap import get_in_activations

from ..dim_reduction_base import DimReductionBase as DRB
from .streaming_pca import StreamingPCA


class IncrementalLDA(StreamingPCA):
    """Supervised reducer that fits an incremental LDA projection."""

    def __init__(self, **kwargs):
        DRB.__init__(self, **kwargs)
        path = Path(kwargs["path"])
        layer = kwargs["layer"]
        model = kwargs["model"]
        dataset = kwargs["dataset"]
        ds_key = kwargs["ds_key"]
        rank = kwargs.get("rank", 300)
        self.cv_dim = kwargs.get("cv_dim", None)
        self.input_key = kwargs.get("input_key", "image")
        self.batch_size = kwargs.get("batch_size", 64)
        self.n_threads = kwargs.get("n_threads", 1)
        self.dataset_portion = kwargs.get("dataset_portion", None)
        self.activations_parser = kwargs.get("activations_parser", get_in_activations)
        self.label_key = kwargs.get("label_key", "label")
        self.n_classes = kwargs.get("n_classes", kwargs.get("nl_model", None))
        self.regularization = kwargs.get("regularization", 1e-6)
        verbose = kwargs.get("verbose", False)

        path.mkdir(parents=True, exist_ok=True)

        self.layer = layer
        self.layer_module = model._target_modules[layer]
        self.device = model.device
        self.file_path = path / f"{layer}.incremental_lda.pt"

        if self.file_path.exists():
            if verbose:
                print(f"File {self.file_path} exists. Loading from disk.")
            self._lda = torch.load(self.file_path, weights_only=False)
        else:
            self._lda = self._fit_from_dataset(
                dataset=dataset,
                ds_key=ds_key,
                model=model,
                rank=rank,
                verbose=verbose,
            )
            if verbose:
                print(f"saving {self.file_path}")
            torch.save(self._lda, self.file_path)

        self.rank = self._lda["components"].shape[0]
        if self.cv_dim is None:
            self.cv_dim = self.rank

        self.class_ids = self._lda["class_ids"].to(dtype=torch.long)
        self.mean = self._lda["mean"].detach().to(self.device)
        self.reduct_m = self._lda["components"].detach().to(self.device)

    def _iter_batches(self, ds):
        n_samples = len(ds)
        for start in range(0, n_samples, self.batch_size):
            stop = min(start + self.batch_size, n_samples)
            yield ds[start:stop]

    def _fit_from_dataset(self, **kwargs):
        dataset = kwargs["dataset"]
        ds_key = kwargs["ds_key"]
        model = kwargs["model"]
        rank = kwargs["rank"]
        verbose = kwargs.get("verbose", False)

        if dataset._dss is None or ds_key not in dataset._dss:
            raise RuntimeError(
                f"Dataset split '{ds_key}' is not available in ParsedDataset._dss. "
                "Load the dataset and keep it open within its context manager before fitting the reducer."
            )

        ds = dataset._dss[ds_key]
        labels = ds[self.label_key].detach().to(dtype=torch.long, device="cpu").view(-1)

        if labels.numel() < 2:
            raise RuntimeError(
                f"Need at least 2 samples to fit incremental LDA on split '{ds_key}'. Got {labels.numel()}."
            )

        class_ids = labels.unique(sorted=True)
        if class_ids.numel() < 2:
            raise RuntimeError(
                f"Need at least 2 classes to fit incremental LDA on split '{ds_key}'. Got {class_ids.numel()}."
            )

        if self.n_classes is not None:
            max_expected = int(self.n_classes) - 1
            if int(class_ids.max().item()) > max_expected:
                raise RuntimeError(
                    f"Found label {int(class_ids.max().item())} but n_classes={self.n_classes}."
                )

        class_to_index = {int(class_id): idx for idx, class_id in enumerate(class_ids.tolist())}
        n_classes = class_ids.numel()

        global_count = 0
        global_mean = None
        class_counts = torch.zeros(n_classes, dtype=torch.long)
        class_means = None
        within_scatter = None

        model.set_activations(save_input=True, save_output=False)
        try:
            with torch.no_grad():
                for batch in self._iter_batches(ds):
                    model(batch[self.input_key].to(model.device))
                    acts = self.activations_parser(model._acts)[self.layer]
                    acts = acts.flatten(start_dim=1).detach().to(dtype=torch.float64, device="cpu")
                    batch_labels = batch[self.label_key].detach().to(dtype=torch.long, device="cpu").view(-1)

                    if acts.shape[0] == 0:
                        continue

                    if global_mean is None:
                        n_features = acts.shape[1]
                        global_mean = torch.zeros(n_features, dtype=torch.float64)
                        class_means = torch.zeros((n_classes, n_features), dtype=torch.float64)
                        within_scatter = torch.zeros((n_features, n_features), dtype=torch.float64)

                    batch_count = acts.shape[0]
                    batch_mean = acts.mean(dim=0)

                    if global_count == 0:
                        global_mean = batch_mean
                        global_count = batch_count
                    else:
                        total_count = global_count + batch_count
                        global_mean = (
                            global_count * global_mean + batch_count * batch_mean
                        ) / total_count
                        global_count = total_count

                    for class_id in batch_labels.unique(sorted=True).tolist():
                        class_mask = batch_labels == class_id
                        acts_class = acts[class_mask]
                        count_batch = acts_class.shape[0]
                        if count_batch == 0:
                            continue

                        batch_class_mean = acts_class.mean(dim=0)
                        centered = acts_class - batch_class_mean
                        scatter_batch = centered.T @ centered

                        class_idx = class_to_index[int(class_id)]
                        count_prev = int(class_counts[class_idx].item())

                        if count_prev == 0:
                            class_means[class_idx] = batch_class_mean
                            class_counts[class_idx] = count_batch
                            within_scatter += scatter_batch
                        else:
                            total_count = count_prev + count_batch
                            delta = batch_class_mean - class_means[class_idx]
                            within_scatter += scatter_batch + (
                                (count_prev * count_batch / total_count) * torch.outer(delta, delta)
                            )
                            class_means[class_idx] = (
                                count_prev * class_means[class_idx] + count_batch * batch_class_mean
                            ) / total_count
                            class_counts[class_idx] = total_count

                    if verbose:
                        print(f"processed {global_count} samples for {self.layer}")
        finally:
            model.set_activations(save_input=False, save_output=False)

        if global_mean is None or within_scatter is None or global_count < 2:
            raise RuntimeError(
                f"Need at least 2 samples to fit incremental LDA on split '{ds_key}'. Got {global_count}."
            )

        valid_classes = class_counts > 0
        class_ids = class_ids[valid_classes]
        class_counts = class_counts[valid_classes]
        class_means = class_means[valid_classes]

        if class_ids.numel() < 2:
            raise RuntimeError(
                f"Need at least 2 populated classes to fit incremental LDA on split '{ds_key}'."
            )

        between_scatter = torch.zeros_like(within_scatter)
        for idx in range(class_ids.numel()):
            delta = class_means[idx] - global_mean
            between_scatter += class_counts[idx].to(dtype=torch.float64) * torch.outer(delta, delta)

        reg_scale = torch.trace(within_scatter) / max(within_scatter.shape[0], 1)
        reg_scale = reg_scale.clamp_min(torch.tensor(self.regularization, dtype=torch.float64))
        sw_reg = within_scatter + self.regularization * reg_scale * torch.eye(
            within_scatter.shape[0], dtype=torch.float64
        )

        sw_evals, sw_evecs = torch.linalg.eigh(sw_reg)
        positive = sw_evals > torch.finfo(sw_evals.dtype).eps
        if not positive.any():
            raise RuntimeError("Within-class scatter is numerically singular after regularization.")

        whitener = sw_evecs[:, positive] / torch.sqrt(sw_evals[positive]).unsqueeze(0)
        whitened_between = whitener.T @ between_scatter @ whitener
        lda_evals, lda_evecs = torch.linalg.eigh(whitened_between)

        order = torch.argsort(lda_evals, descending=True)
        lda_evals = lda_evals[order]
        lda_evecs = lda_evecs[:, order]

        max_rank = min(rank, class_ids.numel() - 1, lda_evecs.shape[1])
        components = (whitener @ lda_evecs[:, :max_rank]).T.contiguous()

        return {
            "class_ids": class_ids.to(dtype=torch.long),
            "mean": global_mean.to(dtype=torch.float32),
            "class_means": class_means.to(dtype=torch.float32),
            "components": components.to(dtype=torch.float32),
            "eigenvalues": lda_evals[:max_rank].clamp_min(0).to(dtype=torch.float32),
            "class_counts": class_counts.to(dtype=torch.long),
            "n_samples": int(global_count),
        }

    def __call__(self, **kwargs):
        act_data = kwargs["act_data"]
        acts_flat = act_data.flatten(start_dim=1)
        centered = acts_flat - self.mean.to(acts_flat.device)
        return centered @ self.reduct_m.to(acts_flat.device).T
