from pathlib import Path
from matplotlib import pyplot as plt

def _get_empp_coverage_scores(**kwargs):
    """
    For each layer, computes:
      - Class coverage: fraction of classes represented by at least one cluster.
      - Cluster coverage: fraction of clusters representing at least one class.

    Returns:
        Tuple of two dicts:
        - class_scores[module] = class_coverage_percent
        - cluster_scores[module] = cluster_coverage_percent
    """
    drillers = kwargs['drillers']
    threshold = kwargs.get('threshold', 0.9) # % to be considered represented

    class_scores = {}
    cluster_scores = {}

    for module, driller in drillers.items():
        empp = driller._empp  # shape: [n_clusters, n_classes]

        # Class coverage
        empp_t = empp.T  # shape: [n_classes, n_clusters]
        class_represented = (empp_t >= threshold).any(dim=1)  # [n_classes]
        n_classes = empp.shape[1]
        n_class_represented = class_represented.sum().item()
        class_coverage = 100 * n_class_represented / n_classes
        class_scores[module] = class_coverage

        # Cluster coverage
        cluster_active = (empp >= threshold).any(dim=1)  # [n_clusters]
        n_clusters = empp.shape[0]
        n_cluster_active = cluster_active.sum().item()
        cluster_coverage = 100 * n_cluster_active / n_clusters
        cluster_scores[module] = cluster_coverage

        print(f"[{module}] Class Coverage: {n_class_represented}/{n_classes} (≥ {threshold}) — {class_coverage:.2f}%")
        print(f"[{module}] Cluster Coverage: {n_cluster_active}/{n_clusters} (≥ {threshold}) — {cluster_coverage:.2f}%")

    return class_scores, cluster_scores

def empp_coverage_scores(**kwargs):
    """
    Computes and (optionally) plots class and cluster coverage per module

    Returns:
        dict: scores[module] = average_coverage
    """
    drillers = kwargs['drillers']
    threshold = kwargs.get('threshold', 0.9) # threshold for coverage
    plot = kwargs.get('plot', False) # if True, plots the coverage
    default_dir = next(iter(drillers.values()))._clas_path.parent
    save_path = kwargs.get('save_path', default_dir) # if plot is True

    # Compute coverage
    class_scores,  cluster_scores= _get_empp_coverage_scores(drillers=drillers, threshold=threshold)

    scores = {}
    for module in class_scores:
        if module in cluster_scores:
            avg = (class_scores[module] + cluster_scores[module]) / 2
            scores[module] = avg
            print(f"[{module}] Avg Coverage: {avg:.2f}% (Class: {class_scores[module]:.2f}%, Cluster: {cluster_scores[module]:.2f}%)")

    # Optional plot
    if plot:
        modules = list(class_scores.keys())
        class_values = [class_scores[m] for m in modules]
        cluster_values = [cluster_scores.get(m, 0) for m in modules]

        plt.figure(figsize=(12, 6))
        plt.plot(modules, class_values, label='Class Coverage', marker='o')
        plt.plot(modules, cluster_values, label='Cluster Coverage', marker='x')

        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Coverage (%)')
        plt.xlabel('Module')
        plt.title(f'Empirical Posterior Coverage (Threshold ≥ {threshold})')
        plt.ylim(0, 105)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_path:
            filename = f'coverage_plot_threshold_{threshold:.2f}.png'
            save_path = Path(save_path)
            path = save_path / filename
            plt.savefig(path, dpi=300)
            print(f"Saved plot to {path}")
        else:
            plt.show()
    return scores

