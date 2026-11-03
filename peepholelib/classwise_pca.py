from pathlib import Path

import torch


class ClasswisePCA:
    """Fit and save one PC1 vector per class at one model layer.

    fit(activations=X, labels=y) accepts [N, D] features or [N, T, D]
    tokens. By default tokens are flattened in order to [N, T * D],
    retaining separate coordinates for each token position and feature.
    n_features defaults to None and is inferred during fit or restored by load.
    An explicit n_features constrains the dimension after token selection.
    Alternatively, fit(model=wrap, datasets=ds) extracts training
    activations using ModelWrap and ParsedDataset.

    save() writes <path>/<layer>/pca_vecs with all class PC1 vectors,
    explained variance and absolute cosine similarities. Diagnostic PNG plots
    are saved alongside the archive.
    Model extraction filters by class before inference and holds only one
    class activation matrix at a time. full_pca=True computes exact PCA;
    the default uses randomized PCA. Only PC1, the mean and diagnostics
    are retained, on CPU.
    """

    def __init__(self, *, path, layer, n_classes, n_features=None, device='cpu',
                 full_pca=False, svd_rank=11, niter=2, seed=0, token_reduction='flatten',
                 activation_source='input', label_key='label'):
        self.path = Path(path)
        self.target_module = layer
        self.n_classes = n_classes
        self.n_features = n_features
        self.device = torch.device(device)
        if type(full_pca) is not bool:
            raise ValueError('full_pca must be a boolean.')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63).')
        for key, value, minimum in (('svd_rank', svd_rank, 1), ('niter', niter, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}.')
        self.full_pca = full_pca
        self.svd_rank = svd_rank
        self.niter = niter
        self.seed = seed
        for key, value in (('n_classes', self.n_classes), ('n_features', self.n_features)):
            if key == 'n_features' and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer.')
        if token_reduction not in ('first', 'mean', 'flatten'):
            raise ValueError('token_reduction must be first, mean, or flatten.')
        if activation_source not in ('input', 'output'):
            raise ValueError('activation_source must be input or output.')
        self.token_reduction = token_reduction
        self.activation_source = activation_source
        self.label_key = label_key
        self.pcas = {}
        self._pca_folder = self.path / self.target_module
        self._pca_path = self._pca_folder / 'pca_vecs'

    def _parse(self, activations):
        if not isinstance(activations, torch.Tensor):
            raise ValueError('Pass a tensor of raw activations.')
        data = activations.detach()
        if data.ndim == 3:
            if self.token_reduction == 'first':
                data = data[:, 0, :]
            elif self.token_reduction == 'mean':
                data = data.mean(dim=1)
            else:
                data = data.flatten(start_dim=1)
        if data.ndim != 2 or data.shape[1] < 1:
            raise ValueError('Expected nonempty features [N, D] after token selection.')
        if self.n_features is not None and data.shape[1] != self.n_features:
            raise ValueError(f'Expected [N, {self.n_features}] after token selection.')
        if not data.is_floating_point() or not torch.isfinite(data).all():
            raise ValueError('Activations must be finite floating-point values.')
        return data

    @staticmethod
    def _labels(dataset, label_key, batch_size):
        """Read only labels where supported; otherwise scan bounded batches."""
        try:
            labels = dataset[label_key]
        except (TypeError, IndexError):
            labels = torch.cat([
                torch.as_tensor(dataset[start:start + batch_size][label_key])
                .detach().cpu().reshape(-1)
                for start in range(0, len(dataset), batch_size)
            ]) if len(dataset) else torch.empty(0, dtype=torch.long)
        return torch.as_tensor(labels).detach().cpu().reshape(-1)

    def _extract(self, model, dataset, indices, input_key, batch_size):
        """Extract only selected class rows, into one preallocated CPU matrix."""
        captured = []

        def hook(module, inputs, output):
            value = inputs[0] if self.activation_source == 'input' else output
            captured.append(self._parse(value).cpu().clone())

        handle = model._target_modules[self.target_module].register_forward_hook(hook)
        modes = [(module, module.training) for module in model.modules()]
        features = None
        try:
            model.eval()
            for start in range(0, len(indices), batch_size):
                selected = indices[start:start + batch_size].tolist()
                batch = dataset[selected]
                captured.clear()
                model(batch[input_key].to(model.device))
                if len(captured) != 1:
                    raise ValueError('Target layer must execute exactly once per batch.')
                value = captured.pop()
                if len(value) != len(selected):
                    raise ValueError('Activation batch size does not match selected images.')
                if features is None:
                    features = torch.empty((len(indices), value.shape[1]), dtype=value.dtype)
                elif value.shape[1] != features.shape[1]:
                    raise ValueError('Activation feature dimensions changed between batches.')
                features[start:start + len(selected)].copy_(value)
        finally:
            handle.remove()
            for module, training in modes:
                module.training = training
        return features

    def _fit_class(self, samples, class_id):
        """Own the class buffer; return compact CPU tensors, never SVD views."""
        dtype = torch.float64 if samples.dtype == torch.float64 else torch.float32
        samples = samples.to(device=self.device, dtype=dtype)
        n, d = samples.shape
        mean = samples.mean(dim=0)
        samples.sub_(mean)
        # Use all centered features for total variance, including in randomized mode.
        total = torch.linalg.vector_norm(samples).square()
        if not torch.isfinite(total) or total <= 0:
            raise ValueError(f'Class {class_id} has invalid or zero variance.')
        if self.full_pca:
            # Reduced factors omit unused bases; no approximation or rank truncation.
            kwargs = {'driver': 'gesvd'} if self.device.type == 'cuda' else {}
            u, singular_values, vh = torch.linalg.svd(samples, full_matrices=False, **kwargs)
        else:
            # Isolate each class's sketch from caller RNG and extraction randomness.
            cuda_index = samples.device.index if samples.is_cuda else None
            devices = [cuda_index] if samples.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                class_seed = (self.seed + class_id) % (2**63)
                torch.random.default_generator.manual_seed(class_seed)
                if samples.is_cuda:
                    with torch.cuda.device(cuda_index):
                        torch.cuda.manual_seed(class_seed)
                u, singular_values, v = torch.pca_lowrank(
                    samples, q=min(self.svd_rank, n - 1, d),
                    center=False, niter=self.niter)
                vh = v.T
        s = singular_values[:1].cpu().clone()
        return {
            'mean': mean.cpu().clone(),
            'components': vh[:1].cpu().clone(),
            'singular_values': s,
            'explained_variance': s.square() / (n - 1),
            'explained_variance_ratio': s.square() / total.cpu(),
            'n_samples': n,
        }

    def _check_fitted(self):
        if not self.pcas:
            raise RuntimeError('PCA is not fitted. Run fit() or load() first.')

    @torch.no_grad()
    def fit(self, *, activations=None, labels=None, model=None, datasets=None,
            loader='train', input_key='image', label_key=None, batch_size=64):
        """Fit one PC1 per class from raw training activations.

        Supply activations/labels OR model/datasets. Each class needs at least
        two observations and nonzero variance. Class IDs are 0..n_classes-1.
        """
        label_key = self.label_key if label_key is None else label_key
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        if activations is not None:
            if labels is None or model is not None or datasets is not None:
                raise ValueError('Supply activations and labels, or model and datasets.')
            if not isinstance(activations, torch.Tensor) or activations.ndim not in (2, 3):
                raise ValueError('Pass activations as [N, D] or [N, T, D].')
            n_samples = len(activations)
            labels = torch.as_tensor(labels).detach().cpu().reshape(-1)
        else:
            if model is None or datasets is None or labels is not None:
                raise ValueError('Supply activations and labels, or model and datasets.')
            dataset = datasets._dss[loader]
            n_samples = len(dataset)
            labels = self._labels(dataset, label_key, batch_size)
        if labels.numel() != n_samples:
            raise ValueError('Labels and activations must have the same sample count.')
        if (not torch.isfinite(labels).all() or (labels != labels.long()).any()
                or (labels < 0).any() or (labels >= self.n_classes).any()):
            raise ValueError('Labels must be integer class IDs in [0, n_classes).')
        counts = torch.bincount(labels.long(), minlength=self.n_classes)
        for class_id, count in enumerate(counts.tolist()):
            if count < 2:
                raise ValueError(f'Class {class_id} needs at least two observations.')

        pcas = {}
        n_features = self.n_features
        for class_id in range(self.n_classes):
            indices = (labels == class_id).nonzero(as_tuple=True)[0]
            if activations is not None:
                # Advanced indexing owns its storage; centering cannot mutate caller data.
                samples = self._parse(activations[indices.to(activations.device)])
            else:
                samples = self._extract(model, dataset, indices, input_key, batch_size)
            if n_features is None:
                n_features = samples.shape[1]
            if samples.shape[1] != n_features:
                raise ValueError('Activation feature dimensions changed between classes.')
            pcas[class_id] = self._fit_class(samples, class_id)
            del samples
        self.pcas = pcas
        self.n_features = n_features
        self.label_key = label_key

    @property
    def pca_vecs(self):
        """One unit PC1 vector per class, in class-ID order: [classes, features]."""
        self._check_fitted()
        return torch.stack([self.pcas[c]['components'][0] for c in range(self.n_classes)])

    @property
    def cosine_similarity_abs(self):
        """Absolute pairwise cosine similarities, invariant to PCA sign flips."""
        vectors = torch.nn.functional.normalize(self.pca_vecs, dim=1)
        return (vectors @ vectors.T).abs().clamp(0, 1)

    def _plots(self, ratios, cosine):
        """Render diagnostic PNGs for saving alongside the vector archive."""
        from io import BytesIO
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.ticker import MaxNLocator, PercentFormatter

        def png(figure):
            buffer = BytesIO()
            FigureCanvasAgg(figure).print_png(buffer)
            return buffer.getvalue()

        figure = Figure(figsize=(max(8, min(20, self.n_classes * .3)), 4),
                        layout='constrained')
        ax = figure.subplots()
        ax.bar(range(self.n_classes), ratios.tolist())
        ax.set(xlabel='Class', ylabel='Variance explained by PC1', ylim=(0, 1),
               title=f'PC1 explained variance by class — {self.target_module}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        explained_png = png(figure)

        figure = Figure(figsize=(7, 6), layout='constrained')
        ax = figure.subplots()
        im = ax.imshow(cosine.numpy(), vmin=0, vmax=1, cmap='viridis',
                       interpolation='nearest')
        ax.set(xlabel='Class', ylabel='Class',
               title=f'Absolute cosine similarity of PC1 — {self.target_module}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
        figure.colorbar(im, ax=ax, label='|cos(PC1_i, PC1_j)|')
        return {'explained_variance_png': explained_png,
                'cosine_similarity_abs_png': png(figure)}

    def save(self):
        """Save tensors/metadata to pca_vecs and diagnostics to separate PNGs.

        All three files live in <path>/<layer>:
        pca_vecs, explained_variance.png, cosine_similarity_abs.png.
        The vector archive is readable with torch.load(weights_only=True).
        """
        self._check_fitted()
        state = {
            'format_version': 8,
            'solver': 'full' if self.full_pca else 'randomized',
            'full_pca': self.full_pca,
            'svd_rank': self.svd_rank,
            'niter': self.niter,
            'seed': self.seed,
            'token_reduction': self.token_reduction,
            'activation_source': self.activation_source,
            'target_module': self.target_module,
            'n_classes': self.n_classes,
            'n_features': self.n_features,
            'label_key': self.label_key,
            'class_ids': torch.arange(self.n_classes),
            'pca_vecs': self.pca_vecs.detach().cpu(),
            'means': torch.stack([self.pcas[c]['mean'] for c in range(self.n_classes)]).cpu(),
            'n_samples': torch.tensor([self.pcas[c]['n_samples'] for c in range(self.n_classes)]),
            'cosine_similarity_abs': self.cosine_similarity_abs.detach().cpu(),
        }
        for key in ('singular_values', 'explained_variance', 'explained_variance_ratio'):
            state[key] = torch.cat([self.pcas[c][key] for c in range(self.n_classes)]).cpu()
        plots = self._plots(state['explained_variance_ratio'],
                            state['cosine_similarity_abs'])
        self._pca_folder.mkdir(parents=True, exist_ok=True)
        torch.save(state, self._pca_path)
        for key, filename in (('explained_variance_png', 'explained_variance.png'),
                              ('cosine_similarity_abs_png', 'cosine_similarity_abs.png')):
            (self._pca_folder / filename).write_bytes(plots[key])
        return self._pca_path

    def load(self):
        """Load all class PC1s from the single layer file; False if absent."""
        if not self._pca_path.exists():
            return False
        state = torch.load(self._pca_path, weights_only=True, map_location='cpu')
        if state.get('format_version') != 8:
            raise ValueError('Legacy PCA cache; refit with ClasswisePCA.')
        expected_solver = 'full' if self.full_pca else 'randomized'
        if state.get('solver') != expected_solver:
            raise ValueError(f'Saved PCA solver does not match {expected_solver!r}.')
        keys = ('target_module', 'n_classes', 'n_features',
                'token_reduction', 'activation_source', 'full_pca')
        if not self.full_pca:
            keys += ('svd_rank', 'niter', 'seed')
        for key in keys:
            if key == 'n_features' and self.n_features is None:
                continue
            if state[key] != getattr(self, key):
                raise ValueError(f'Saved PCA {key}={state[key]!r} does not match {getattr(self, key)!r}.')
        saved_features = state['n_features']
        if type(saved_features) is not int or saved_features < 1:
            raise ValueError('Saved PCA n_features must be a positive integer.')
        if (state['pca_vecs'].ndim != 2
                or state['pca_vecs'].shape[1] != saved_features
                or state['means'].shape != state['pca_vecs'].shape):
            raise ValueError('Saved PCA dimensions do not match n_features.')
        pcas = {}
        for c in range(self.n_classes):
            pcas[c] = {
                'mean': state['means'][c].clone(),
                'components': state['pca_vecs'][c:c+1].clone(),
                'n_samples': state['n_samples'][c].item(),
            }
            for key in ('singular_values', 'explained_variance', 'explained_variance_ratio'):
                pcas[c][key] = state[key][c:c+1].clone()
        self.pcas = pcas
        self.n_features = saved_features
        self.label_key = state['label_key']
        return True
