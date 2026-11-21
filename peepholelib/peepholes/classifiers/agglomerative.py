"""SciPy hierarchical clustering of class PC1 vectors or labeled feature rows."""
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import csv

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram
from scipy.spatial.distance import cdist, squareform

from .classifier_base import ClassifierBase
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score

def _identity(*, cvs):
    return cvs


class Agglomerative(ClassifierBase):
    """Ward linkage with sign-invariant projection chordal distance by default.

    For unit PC1s u, v, chordal distance is sqrt(1 - (u.T @ v)**2),
    the Euclidean distance between uu.T / sqrt(2) and vv.T / sqrt(2).
    Other linkage methods default to absolute_cosine distance.

    Uses the TGMM driller interface and saves under path/name. Supply
    nl_classifier for the final cut. fit accepts data/labels, pca_path (a
    ClasswisePCA layer directory or archive), or datasets/corevectors.
    Each input row counts once, including one row per class in pca_vecs.
    Level 0 has N singleton clusters; level L follows L merges (N-L clusters).
    label_names accepts an ID-to-name mapping or 'cifar100' for standard fine
    labels. save() writes one PDF page per level, with t-SNE, class-name key,
    population bars, silhouette and stability (mean pairwise resampling ARI);
    silhouette_scores.csv stores both scores and their harmonic mean.
    PCA fitting computes stability by
    default: supply feature_provider(class_id) returning original activations,
    or model/dataset for flattened raw layer inputs. It uses 20 subsets of 80%
    of each class's images, retaining every class. stability_options configures
    repeats, sample_fraction, seed, device and scratch. stability=False opts out.
    Generic data/labels fitting computes stability when a source is supplied.
    Without resampling results, reports explicitly show stability as N/A.

    SciPy has no out-of-sample predict: this adapter assigns new rows to the
    cluster of their nearest training vector. Probabilities are hard one-hot
    assignments, not calibrated likelihoods. Clustering and plotting use CPU.
    """

    def __init__(self, *, method='ward', metric=None,
                 plot=True, random_state=0, label_names=None, **kwargs):
        kwargs.setdefault('reducer', SimpleNamespace(parser=_identity))
        super().__init__(**kwargs)
        self.device = torch.device(self.device)
        if method not in ('single', 'complete', 'average', 'weighted', 'ward'):
            raise ValueError('Unsupported linkage method.')
        if metric is None:
            metric = 'chordal' if method == 'ward' else 'absolute_cosine'
        if metric not in ('absolute_cosine', 'chordal', 'euclidean'):
            raise ValueError('metric must be absolute_cosine, chordal or euclidean.')
        if method == 'ward' and metric == 'absolute_cosine':
            raise ValueError('Ward requires Euclidean distances; use chordal for sign-invariant PC1s.')
        self.method, self.metric = method, metric
        self.plot, self.random_state = plot, random_state
        if label_names == 'cifar100':
            from .class_names import CIFAR100_LABEL_NAMES
            if self.nl_model != 100:
                raise ValueError('CIFAR-100 names require nl_model=100.')
            label_names = CIFAR100_LABEL_NAMES
        self.label_names = {int(k): str(v) for k, v in (label_names or {}).items()}
        self._clas_path = Path(self.path) / self.name
        self._empp_file = self._clas_path / f'empp_{self.label_key}.pt'
        self._state_file = self._clas_path / 'agglomerative.pt'
        self.linkage_matrix = None
        self.stability_ = None

    @classmethod
    def from_pca(cls, pca_path, *, path, name, nl_classifier,
                 feature_provider=None, model=None, dataset=None, stability=True,
                 stability_options=None, **kwargs):
        """Fit a PCA archive and resample its source images for stability by default.

        Supply feature_provider or model/dataset, or explicitly stability=False.
        PCA vectors alone do not contain the image variation needed for ARI.
        """
        state = cls._read_pca(pca_path)
        obj = cls(path=path, name=name, nl_classifier=nl_classifier,
                  target_module=state['target_module'], nl_model=state['n_classes'],
                  n_features=state['n_features'], **kwargs)
        obj.fit(data=state['pca_vecs'], labels=state['class_ids'],
                feature_provider=feature_provider, model=model, dataset=dataset,
                stability=stability, stability_options=stability_options)
        return obj

    @staticmethod
    def _read_pca(path):
        path = Path(path)
        return torch.load(path / 'pca_vecs' if path.is_dir() else path,
                          weights_only=True, map_location='cpu')

    def _data(self, data):
        data = torch.as_tensor(data).detach().cpu()
        if (data.ndim != 2 or data.shape[1] != self.n_features
                or not data.is_floating_point() or not torch.isfinite(data).all()):
            raise ValueError(f'Expected finite floating-point rows [N, {self.n_features}].')
        data = data.to(torch.float64 if data.dtype == torch.float64 else torch.float32)
        if self.metric in ('absolute_cosine', 'chordal'):
            if self.metric == 'chordal':
                data = data.double()
            norms = torch.linalg.vector_norm(data, dim=1, keepdim=True)
            if (norms == 0).any():
                raise ValueError('Directional distance is undefined for zero vectors.')
            data = data / norms
        return data

    def _distances(self, a, b):
        if self.metric == 'chordal':
            cosine = (a @ b.T).clamp(-1, 1)
            return (1 - cosine.square()).clamp_min(0).sqrt().numpy().astype(np.float64)
        if self.metric == 'absolute_cosine':
            return (1 - (a @ b.T).abs().clamp(0, 1)).numpy().astype(np.float64)
        return cdist(a.numpy(), b.numpy(), metric='euclidean')

    def _check_fitted(self):
        if self.linkage_matrix is None:
            raise RuntimeError('Run fit() or load() first.')

    def fit(self, **kwargs):
        feature_provider = kwargs.get('feature_provider')
        model, dataset = kwargs.get('model'), kwargs.get('dataset')
        stability = kwargs.get('stability', 'pca_path' in kwargs or
                               feature_provider is not None or model is not None or dataset is not None)
        if stability and feature_provider is None and (model is None or dataset is None):
            raise ValueError('Stability requires feature_provider or model and dataset; '
                             'pass stability=False to fit without resampling.')
        if 'pca_path' in kwargs:
            state = self._read_pca(kwargs['pca_path'])
            if state['target_module'] != self.target_module:
                raise ValueError('PCA target layer does not match the driller.')
            data, labels = state['pca_vecs'], state['class_ids']
        elif 'data' in kwargs:
            data, labels = kwargs['data'], kwargs['labels']
        else:
            loader = kwargs.get('loader', 'train')
            data = self.parser(cvs=kwargs['corevectors']._corevds[loader][self.target_module])
            labels = kwargs['datasets']._dss[loader][:][self.label_key]
        data = self._data(data)
        labels = torch.as_tensor(labels).detach().cpu().reshape(-1)
        n = len(data)
        if n < 2 or type(self.nl_class) is not int or not 1 <= self.nl_class <= n:
            raise ValueError('Need at least two rows and 1 <= nl_classifier <= N.')
        if (len(labels) != n or not torch.isfinite(labels).all()
                or (labels != labels.long()).any() or (labels < 0).any()
                or (labels >= self.nl_model).any()):
            raise ValueError('Need one integer label in [0, nl_model) per row.')
        distances = self._distances(data, data)
        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0)
        tree = linkage(squareform(distances, checks=False), method=self.method)
        assignments = cut_tree(tree, n_clusters=[self.nl_class])[:, 0]
        self.stability_ = None
        self.points_, self.labels_ = data, labels.long()
        self.linkage_matrix = tree
        self.cluster_labels_ = torch.from_numpy(assignments.copy()).long()
        self._classifier = tree
        self._empp = None
        if kwargs.get('compute_empp', True):
            counts = torch.bincount(self.cluster_labels_ * self.nl_model + self.labels_,
                                   minlength=self.nl_class * self.nl_model)
            counts = counts.reshape(self.nl_class, self.nl_model).float()
            self._empp = (counts / counts.sum(1, keepdim=True)).to(self.device)
        if stability:
            self.compute_stability(feature_provider, model=model, dataset=dataset,
                                   **(kwargs.get('stability_options') or {}))
        return self

    def compute_stability(self, feature_provider=None, *, model=None, dataset=None,
                          batch_size=64, repeats=20, sample_fraction=.8,
                          seed=29, device='cpu', scratch=None):
        """Extract one class at a time; spill replicate PC1s to temporary disk.

        Original PC1s are checked against the supplied feature extraction before
        accepting results. Only compact distances/cuts/ARI samples are retained.
        Pairwise quantiles describe a distribution, not independent confidence bounds.
        """
        self._check_fitted()
        self.stability_ = None
        if feature_provider is not None and (model is not None or dataset is not None):
            raise ValueError('Supply feature_provider OR model and dataset.')
        if feature_provider is None:
            if model is None or dataset is None:
                raise ValueError('Stability requires feature_provider or model and dataset; '
                                 'saved PC1 vectors alone cannot be resampled.')
            labels = read_labels(dataset, label_key=self.label_key)
            if len(labels) != len(dataset):
                raise ValueError('Dataset labels do not match its image count.')
            def feature_provider(class_id):
                indices = (labels == class_id).nonzero(as_tuple=True)[0].tolist()
                return extract_class_features(model, self.target_module, dataset, indices,
                                              batch_size=batch_size, device=device)
        if repeats < 2 or not 0 < sample_fraction < 1:
            raise ValueError('Need repeats >= 2 and 0 < sample_fraction < 1.')
        if self.metric not in ('absolute_cosine', 'chordal'):
            raise ValueError('Class PC1 stability requires sign-invariant absolute_cosine or chordal distance.')
        if self.method == 'ward' and self.metric != 'chordal':
            raise ValueError('Ward PC1 stability requires projection chordal distance.')
        ids = self.labels_.tolist()
        if len(ids) != len(set(ids)) or len(ids) < 3:
            raise ValueError('Need one PC1 per class, and at least three classes.')
        n, d = self.points_.shape
        counts, kept_counts, alignments = [], [], []
        distances = np.empty((repeats, n, n), dtype=np.float64)
        cuts = np.empty((repeats, n, n), dtype=np.int64)
        if scratch is not None:
            Path(scratch).mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='pc1-resampling-', dir=scratch) as temporary:
            vectors = np.lib.format.open_memmap(Path(temporary) / 'vectors.npy', mode='w+',
                                               dtype=np.float32, shape=(repeats, n, d))
            for position, class_id in enumerate(ids):
                print(f'  Refit class {class_id} ({position + 1}/{n}), {repeats} subsets', flush=True)
                features = feature_provider(class_id).to(device)
                if features.shape[1] != d:
                    raise ValueError('Raw activation dimensions do not match the original PCA.')
                full, sampled, indices = subsample_pc1s(features, repeats=repeats,
                                                      sample_fraction=sample_fraction, seed=seed + class_id)
                expected = self.points_[position].to(full)
                alignment = float(torch.dot(full, expected / expected.norm()).abs())
                if alignment < .999:
                    raise ValueError(f'Class {class_id}: full-data PC1 alignment {alignment:.6f} < .999. '
                                     'Check checkpoint, preprocessing, dataset and PCA configuration '
                                     '(or a nearly degenerate leading eigenspace).')
                counts.append(len(features))
                kept_counts.append(indices.shape[1])
                alignments.append(alignment)
                vectors[:, position] = sampled.cpu().float().numpy()
                del features, full, sampled, indices
            vectors.flush()
            for run in range(repeats):
                # Only one run's class vectors is materialized at a time.
                # Reuse the classifier's normalization and geometry, including
                # projection chordal distances for Ward.
                points = self._data(torch.from_numpy(np.asarray(vectors[run])))
                matrix = self._distances(points, points)
                matrix = (matrix + matrix.T) / 2
                np.fill_diagonal(matrix, 0)
                distances[run] = matrix
                cuts[run] = cut_tree(linkage(squareform(matrix, checks=False), method=self.method))
                del points
            del vectors
        print('  Comparing resampled partitions at every nontrivial level', flush=True)
        samples = pairwise_ari(cuts)
        self.stability_ = {
            'ari_samples': samples,
            'mean_ari': samples.mean(dim=0),
            'replicate_cuts': torch.from_numpy(cuts),
            'replicate_distances': torch.from_numpy(distances),
            'metadata': {
                'repeats': repeats, 'sample_fraction': sample_fraction, 'seed': seed,
                'class_ids': ids, 'class_counts': counts, 'retained_counts': kept_counts,
                'full_pc1_alignments': alignments, 'method': self.method,
                'metric': self.metric, 'sampling': 'within_class_without_replacement',
                'ari_comparison': 'all_pairs_of_resampled_partitions',
                'average_formula': '(silhouette + mean_ari) / 2',
            },
        }
        return self.stability_

    def predict(self, data):
        """Assign by nearest fitted vector (an extension of the fitted hierarchy)."""
        self._check_fitted()
        data = self._data(data)
        result = []
        for batch in data.split(256):
            nearest = self._distances(batch, self.points_).argmin(axis=1)
            result.append(self.cluster_labels_[nearest])
        return (torch.cat(result) if result else torch.empty(0, dtype=torch.long)).to(self.device)

    def classifier_probabilities(self, **kwargs):
        return torch.nn.functional.one_hot(self.predict(kwargs['data']), self.nl_class).float()

    def level_assignments(self):
        """Columns are levels 0..N-1, including leaves and root."""
        self._check_fitted()
        return cut_tree(self.linkage_matrix)

    def _label(self, value):
        return self.label_names.get(int(value), str(int(value)))

    def save_plots(self, *, levels=None):
        """Save the hierarchy and exactly one PDF page per requested merge level.

        Each page contains a fixed t-SNE embedding colored by that level's
        clusters, the class-name key, cluster populations, and silhouette computed
        in the original distance geometry. CIFAR-100 keys use 20 superclasses.
        Requested levels with the highest silhouette or harmonic mean of
        silhouette and stability have red headings (including ties). The harmonic
        mean is 2*s*a/(s+a) for nonnegative finite scores, or zero if both are zero;
        negative or unavailable inputs are reported as N/A. Endpoint levels
        have undefined silhouette and stability, reported as N/A (blank in
        the CSV). Missing resampling results are explicitly marked not computed.
        """

        self._check_fitted()
        self._clas_path.mkdir(parents=True, exist_ok=True)
        n = len(self.points_)
        levels = list(range(n)) if levels is None else list(levels)
        if any(type(level) is not int or not 0 <= level < n for level in levels):
            raise ValueError('levels must be integer merge steps in [0, N).')
        if len(set(levels)) != len(levels):
            raise ValueError('levels must not contain duplicates.')
        fig = Figure(figsize=(max(10, min(30, n * .15)), 6), layout='constrained')
        ax = fig.subplots()
        dendrogram(self.linkage_matrix, ax=ax,
                   labels=[self._label(x) for x in self.labels_], leaf_rotation=90)
        ax.set(title=f'Hierarchical clustering — {self.target_module}',
               ylabel=f'{self.method} linkage distance ({self.metric})', xlabel='Class vector')
        fig.savefig(self._clas_path / 'hierarchy.png', dpi=150)
        distances = self._distances(self.points_, self.points_)
        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0)
        embedding = TSNE(n_components=2, metric='precomputed', init='random',
                         perplexity=min(30., n-1.), learning_rate='auto',
                         random_state=self.random_state).fit_transform(distances)
        np.save(self._clas_path / 'tsne_coordinates.npy', embedding)
        cuts = self.level_assignments()
        class_ids, label_indices = np.unique(self.labels_.numpy(), return_inverse=True)
        level_scores = {
            level: silhouette_score(distances, cuts[:, level], metric='precomputed')
            if 1 < len(np.unique(cuts[:, level])) < n else None
            for level in levels
        }
        best_score = max((s for s in level_scores.values() if s is not None), default=None)
        harmonic_scores = {}
        for level, score in level_scores.items():
            stability = (float(self.stability_['mean_ari'][level])
                         if self.stability_ is not None else None)
            harmonic = None
            if (score is not None and stability is not None
                    and np.isfinite(score) and np.isfinite(stability)
                    and score >= 0 and stability >= 0):
                harmonic = (2 * score * stability / (score + stability)
                            if score + stability > 0 else 0.)
            harmonic_scores[level] = harmonic
        best_harmonic = max((s for s in harmonic_scores.values() if s is not None),
                            default=None)
        from .class_names import CIFAR100_LABEL_NAMES, CIFAR100_SUPERCLASSES
        is_cifar100 = self.nl_model == 100 and self.label_names == CIFAR100_LABEL_NAMES
        with PdfPages(self._clas_path / 'cluster_populations.pdf') as pdf, \
                (self._clas_path / 'cluster_populations.csv').open('w', newline='') as output, \
                (self._clas_path / 'silhouette_scores.csv').open('w', newline='') as scores:
            writer = csv.writer(output)
            writer.writerow(['level', 'n_clusters', 'cluster', 'population',
                             'label_id', 'label_name', 'count', 'percentage'])
            score_writer = csv.writer(scores)
            score_writer.writerow(['level', 'n_clusters', 'silhouette_score', 'status',
                                   'stability_mean_ari', 'stability_status', 'harmonic_mean'])
            for level in levels:
                assignments = cuts[:, level]
                k = len(np.unique(assignments))
                score = level_scores[level]
                reason = 'defined' if score is not None else (
                    'undefined: one cluster' if k == 1 else 'undefined: all singleton clusters')
                stability = None
                stability_reason = reason if score is None else 'not computed'
                if score is not None and self.stability_ is not None:
                    stability = float(self.stability_['mean_ari'][level])
                    stability_reason = 'defined'
                harmonic = harmonic_scores[level]
                score_writer.writerow([level, k, score, reason, stability, stability_reason,
                                       harmonic])
                counts = np.zeros((len(class_ids), k), dtype=int)
                np.add.at(counts, (label_indices, assignments), 1)
                populations = counts.sum(axis=0)
                percentages = 100 * counts / populations[None, :]
                for j, label in enumerate(class_ids):
                    for cluster in range(k):
                        writer.writerow([level, k, cluster, populations[cluster], int(label),
                                         self._label(label), counts[j, cluster], percentages[j, cluster]])
                colors = colormaps['turbo'].resampled(max(k, 2))(np.arange(k))
                fig = Figure(figsize=(20, 12))
                best = score is not None and score == best_score
                best_combined = harmonic is not None and harmonic == best_harmonic
                score_text = f'{score:.4f}' if score is not None else f'N/A ({reason.removeprefix("undefined: ")})'
                stability_text = (f'{stability:.4f}' if stability is not None else
                                  f'N/A ({stability_reason.removeprefix("undefined: ")})')
                fig.text(.5, .97,
                         f'Level {level}: {k} clusters | Silhouette: {score_text}'
                         f' | Stability (ARI): {stability_text}'
                         + (' | Highest silhouette' if best else ''),
                         ha='center', va='top', fontsize=13,
                         color='red' if best or best_combined else 'black', weight='bold')
                harmonic_text = f'{harmonic:.4f}' if harmonic is not None else 'N/A'
                fig.text(.5, .94,
                         f'{self.target_module} | Silhouette distance: {self.metric}'
                         f' | Harmonic mean (silhouette, stability): {harmonic_text}'
                         + (' | Highest harmonic mean' if best_combined else ''),
                         ha='center', va='top', fontsize=10,
                         color='red' if best_combined else 'black')
                ax = fig.add_axes([.05, .38, .36, .51])
                ax.scatter(*embedding.T, color=colors[assignments], s=35)
                for xy, label in zip(embedding, self.labels_):
                    ax.annotate(str(int(label)), xy, fontsize=6, xytext=(3, 3), textcoords='offset points')
                ax.set(title='t-SNE (fixed coordinates; color = cluster)',
                       xlabel='t-SNE 1', ylabel='t-SNE 2')
                key = fig.add_axes([.45, .36, .53, .53])
                key.axis('off')
                key.set_title('Class number → class name'
                              + (' — CIFAR-100 superclasses' if is_cifar100 else ''),
                              loc='left', fontsize=11, pad=14)
                if is_cifar100:
                    for i, (superclass, members) in enumerate(CIFAR100_SUPERCLASSES.items()):
                        col, row = divmod(i, 7)
                        x, y = col / 3, 1 - row / 7
                        key.text(x, y, superclass, fontsize=8, weight='bold', va='top')
                        key.text(x, y - .025,
                                 '\n'.join(f'{label}: {self._label(label)}' for label in members),
                                 fontsize=8, va='top', linespacing=1.15)
                else:
                    key_columns = min(3, max(1, int(np.ceil(len(class_ids) / 35))))
                    key_rows = int(np.ceil(len(class_ids) / key_columns))
                    for i, label in enumerate(class_ids):
                        col, row = divmod(i, key_rows)
                        name = self.label_names.get(int(label), '(name unavailable)')
                        key.text(col / key_columns, 1 - (row + .5) / key_rows,
                                 f'{label}: {name}', fontsize=9, va='center')
                population_ax = fig.add_axes([.05, .08, .92, .21])
                bars = population_ax.bar(np.arange(k), populations, color=colors)
                population_ax.bar_label(bars, fontsize=7, padding=3)
                population_ax.set(xticks=np.arange(k), xlabel='Cluster',
                                  ylabel='Population (input vectors)',
                                  title='Population of each cluster',
                                  xlim=(-.7, k - .3), ylim=(0, populations.max() * 1.2))
                population_ax.tick_params(axis='x', labelsize=7, rotation=90 if k > 40 else 0)
                population_ax.spines[['top', 'right']].set_visible(False)
                pdf.savefig(fig)
                fig.clear()

    def save(self, **kwargs):
        self._check_fitted()
        self._clas_path.mkdir(parents=True, exist_ok=True)
        state = {key: getattr(self, key) for key in
                 ('target_module', 'nl_model', 'n_features', 'nl_class', 'label_key',
                  'method', 'metric', 'random_state', 'label_names', 'points_',
                  'labels_', 'cluster_labels_')}
        state['stability'] = self.stability_
        state['linkage_matrix'] = torch.from_numpy(self.linkage_matrix)
        state['empp'] = None if self._empp is None else self._empp.cpu()
        torch.save(state, self._state_file)
        super().save()
        if kwargs.get('plot', self.plot):
            self.save_plots(levels=kwargs.get('levels'))
        return self._clas_path

    def load(self, **kwargs):
        if not self._state_file.exists():
            return False
        state = torch.load(self._state_file, weights_only=True, map_location='cpu')
        for key in ('target_module', 'nl_model', 'n_features', 'nl_class', 'method', 'metric', 'label_key'):
            if state[key] != getattr(self, key):
                raise ValueError(f'Saved {key} does not match this driller.')
        for key in ('points_', 'labels_', 'cluster_labels_', 'random_state', 'label_names'):
            setattr(self, key, state[key])
        self.stability_ = state.get('stability')
        self.linkage_matrix = state['linkage_matrix'].numpy()
        self._classifier = self.linkage_matrix
        self._empp = None if state['empp'] is None else state['empp'].to(self.device)
        return True


def _pc1_coefficients(gram, indices):
    """Dual exact PCA: center the selected Gram matrix, then eigendecompose."""
    selected = gram[indices][:, indices]
    centered = selected - selected.mean(0, keepdim=True) - selected.mean(1, keepdim=True) + selected.mean()
    centered = (centered + centered.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(centered)
    value = eigenvalues[-1]
    if not torch.isfinite(value) or value <= 0:
        raise ValueError('A PCA subset has zero or nonfinite variance.')
    weights = eigenvectors[:, -1]
    weights = (weights - weights.mean()) / value.sqrt()
    coefficients = torch.zeros(len(gram), dtype=gram.dtype, device=gram.device)
    coefficients[indices] = weights
    return coefficients


@torch.no_grad()
def subsample_pc1s(features, *, repeats=20, sample_fraction=.8, seed=29):
    """Return full-data PC1 and exact subset PC1s, without changing features.

    A single Gram matrix is reused; each subset has its own centering and exact
    eigendecomposition. This is algebraically equivalent to full centered SVD,
    not randomized/truncated PCA. No images are duplicated within a subset.
    """
    if repeats < 2 or not 0 < sample_fraction < 1:
        raise ValueError('Need repeats >= 2 and 0 < sample_fraction < 1.')
    if features.ndim != 2 or not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError('Expected finite floating-point activations [images, features].')
    n = len(features)
    m = int(n * sample_fraction)
    if not 2 <= m < n:
        raise ValueError(f'Subsampling {n} images at {sample_fraction} must retain 2..N-1 images.')
    centered = features - features.mean(0)
    # Preserve input precision for the large matrix multiply; solve the small
    # symmetric eigenproblem in float64 for more reliable eigenvectors.
    gram = (centered @ centered.T).double()
    generator = torch.Generator().manual_seed(seed)
    subsets = torch.stack([torch.randperm(n, generator=generator)[:m] for _ in range(repeats)])
    weights = [_pc1_coefficients(gram, torch.arange(n, device=gram.device))]
    weights.extend(_pc1_coefficients(gram, indices.to(gram.device)) for indices in subsets)
    vectors = torch.stack(weights).to(centered.dtype) @ centered
    norms = vectors.norm(dim=1, keepdim=True)
    if not torch.isfinite(vectors).all() or (norms <= 0).any():
        raise ValueError('Could not obtain finite nonzero PC1s.')
    vectors = vectors / norms
    return vectors[0], vectors[1:], subsets


def pairwise_ari(cuts):
    """cuts[run, class, level] -> ARI[run-pair, level], endpoints undefined."""
    cuts = np.asarray(cuts)
    if cuts.ndim != 3 or cuts.shape[0] < 2 or cuts.shape[1] != cuts.shape[2]:
        raise ValueError('Expected cuts [at least two runs, classes, merge levels].')
    repeats, n, _ = cuts.shape
    samples = np.full((repeats * (repeats - 1) // 2, n), np.nan)
    row = 0
    for first in range(repeats):
        for second in range(first):
            for level in range(1, n - 1):
                samples[row, level] = adjusted_rand_score(cuts[first, :, level], cuts[second, :, level])
            row += 1
    return torch.from_numpy(samples)


@torch.no_grad()
def extract_class_features(model, layer, dataset, indices, *, batch_size, device):
    """One model pass per image, collecting raw layer inputs for one class."""
    captured = []
    def hook(module, inputs):
        captured.append(inputs[0].detach().flatten(1).cpu().clone())
    if batch_size < 1 or len(indices) < 2:
        raise ValueError('Need batch_size >= 1 and at least two images per class.')
    module = getattr(model, '_model', model).get_submodule(layer)
    handle = module.register_forward_pre_hook(hook)
    modes = [(module, module.training) for module in model.modules()]
    features = None
    try:
        model.eval()
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            captured.clear()
            model(dataset[selected]['image'].to(device))
            if len(captured) != 1:
                raise ValueError('Target layer must execute once per batch.')
            batch = captured.pop()
            if features is None:
                features = torch.empty((len(indices), batch.shape[1]), dtype=batch.dtype)
            features[start:start + len(selected)] = batch
    finally:
        handle.remove()
        for module, training in modes:
            module.training = training
    return features


def read_labels(dataset, label_key='label'):
    """Read shard labels without transforming all images just to obtain labels."""
    original = getattr(dataset, 'ori', dataset)
    shards = getattr(original, 'shards', None)
    if shards is not None:
        return torch.cat([torch.as_tensor(shard[label_key]).cpu().reshape(-1) for shard in shards])
    try:
        return torch.as_tensor(dataset[label_key]).cpu().reshape(-1)
    except (TypeError, IndexError, ValueError):
        return torch.cat([torch.as_tensor(dataset[start:start + 64][label_key]).cpu().reshape(-1)
                          for start in range(0, len(dataset), 64)])
