import os

import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

def plot_empp_posteriors(**kwargs):
    """
    For each layer in `drillers`, saves a heatmap of the empirical posterior P(g|c).
    Files are saved as `<name>_<module>.jpeg`.
    """
    drillers = kwargs["drillers"] # already loaded
    save_dir = kwargs.get("save_dir", ".")
    name = kwargs.get("name", "empp")

    os.makedirs(save_dir, exist_ok=True)

    saved = {}
    skipped = []

    for module, driller in drillers.items():
        empp = driller._empp  # torch tensor: [n_clusters, n_classes]
        if empp is None:
            skipped.append((module, "empirical posterior is None"))
            continue

        if hasattr(empp, "detach"):
            empp_np = empp.detach().float().cpu().numpy()
        else:
            empp_np = empp

        try:
            empp_np = np.asarray(empp_np, dtype=float)
        except (TypeError, ValueError) as exc:
            skipped.append((module, f"non-numeric empirical posterior ({type(empp).__name__}): {exc}"))
            continue

        empp_np = np.atleast_2d(empp_np)
        if empp_np.ndim != 2 or empp_np.size == 0:
            skipped.append((module, f"invalid empirical posterior shape {empp_np.shape}"))
            continue

        if not np.isfinite(empp_np).any():
            skipped.append((module, "empirical posterior contains no finite values"))
            continue

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(
            empp_np,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )

        _, n_concepts = empp_np.shape
        concept_ticks = np.arange(n_concepts)
        ax.set_xticks(concept_ticks)
        ax.set_xticklabels(concept_ticks + 1)

        ax.set_xlabel("Concept Index")
        ax.set_ylabel("Cluster Index")

        ax.set_title("%s - %s" % ("Empirical Posterior", module))

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("P(g|c)")

        safe_module = str(module).replace("/", "_").replace(" ", "_")
        safe_name = str(name).replace("/", "_").replace(" ", "_")
        save_path = os.path.join(save_dir, "%s_%s.%s" % (safe_name, safe_module, "jpeg"))

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

        saved[module] = save_path
        print("Empp saved:", save_path)

    if skipped:
        for module, reason in skipped:
            print(f"Skipping module '{module}': {reason}.")

    if not saved:
        raise RuntimeError(
            "No valid empirical posterior matrices were found to plot. "
            "Ensure each driller has `_empp` computed/loaded before calling plot_empp_posteriors."
        )

    return saved

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
       # empp = driller["_empp"]
        empp = driller._empp  # shape: [n_clusters, n_classes]
        # Class coverage
        empp_t = empp.T  # shape: [n_classes, n_clusters]
        class_represented = (empp_t >= threshold).any(dim=1)  # [n_classes]
        n_classes = empp.shape[1]
        n_class_represented = class_represented.sum().item()
        class_coverage =  n_class_represented / n_classes
        class_scores[module] = class_coverage

        # Cluster coverage
        cluster_active = (empp >= threshold).any(dim=1)  # [n_clusters]
        n_clusters = empp.shape[0]
        n_cluster_active = cluster_active.sum().item()
        cluster_coverage = n_cluster_active / n_clusters
        cluster_scores[module] = cluster_coverage

        print(f"[{module}] Class Coverage: {n_class_represented}/{n_classes} (≥ {threshold}) — {class_coverage:.2f}%")
        print(f"[{module}] Cluster Coverage: {n_cluster_active}/{n_clusters} (≥ {threshold}) — {cluster_coverage:.2f}%")

    return class_scores, cluster_scores


def empp_relative_coverage_scores(**kwargs):
    """
    Computes and (optionally) plots the relative coverage:
        relative_coverage = average_coverage / n_clusters

    Where:
        average_coverage = (class_coverage + cluster_coverage) / 2

    Returns:
        dict: relative_scores[module] = relative_coverage
    """
    drillers = kwargs['drillers']
    threshold = kwargs.get('threshold', 0.9)
    plot = kwargs.get('plot', False)

    save_path = kwargs.get('save_path', None)
    filename = kwargs.get('file_name', None)

    # get coverage scores
    class_scores, cluster_scores = _get_empp_coverage_scores(
        drillers=drillers,
        threshold=threshold
    )

    # compute relative scores
    relative_scores = {}

    for module in class_scores:
        if module not in cluster_scores:
            continue

        avg_cov = (class_scores[module] + cluster_scores[module]) / 2

        n_clusters = drillers[module].nl_class
        relative = avg_cov / n_clusters
        #relative = cluster_scores[module] / n_clusters
        relative_scores[module] = relative

        print(
            f"[{module}] Relative Coverage: {relative:.6f} "
            #f"(AvgCov: {avg_cov:.2f}%, n_clusters: {n_clusters})"
        )

    # Optional plotting 
    if plot:
        import matplotlib.pyplot as plt
        from pathlib import Path

        modules = list(relative_scores.keys())
        rel_values = [relative_scores[m] for m in modules]

        plt.figure(figsize=(12, 6))
        plt.plot(modules, rel_values, label='Relative Coverage', marker='o')

        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Relative Coverage (AvgCov / n_clusters)')
        plt.xlabel('Module')
        plt.title(f'Relative Empirical Posterior Coverage (Threshold ≥ {threshold})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save or show
        if save_path:
            if filename is None:
                filename = f'relative_coverage_plot_threshold_{threshold:.2f}.png'

            save_path = Path(save_path)
            out_path = save_path / filename
            plt.savefig(out_path, dpi=300)
            print(f"Saved relative coverage plot to {out_path}")
        else:
            plt.show()

    return relative_scores

def compare_relative_coverage_all_clusters(**kwargs):
    """
    Compare relative coverage across multiple cluster configurations.

    Args:
        all_drillers: all_drillers[n_cluster] = drillers (see function load_all_drillers in xp_coverage_vgg.py)
        threshold: Threshold for coverage computation. Ex: 0.9 means a class is represented if any P(g|c) ≥ 0.9.
        plot: If True, plots the relative coverage across cluster settings.
        save_path: Directory to save plots (if plot=True). If None, shows plots instead
        file_name: Filename for saving the plot (if plot=True). If None, a default name is used.

    Returns:
        all_results[n_clusters][module] = relative_coverage_score
    """

    all_drillers = kwargs["all_drillers"]     
    threshold = kwargs.get("threshold", 0.9)
    plot = kwargs.get("plot", False)
    save_path = kwargs.get("save_path", None)
    filename = kwargs.get("filename", None)

    all_results = {}

    # Loop over all cluster settings
    for n_clusters, drillers in all_drillers.items():

        if not drillers:
            print(f"No drillers found for n_clusters={n_clusters}")
            continue

        print(f"Processing driller set for n_clusters={n_clusters}.")

        # Compute relative coverage for this configuration
        relative_scores = empp_relative_coverage_scores(
            drillers=drillers,
            threshold=threshold,
            plot=False
        )

        all_results[n_clusters] = relative_scores

    # Collect full module list across all configs

    all_modules = []
    for res in all_results.values():
        for m in res.keys():
            if m not in all_modules:
                all_modules.append(m)
    # Plot results
    if plot:
        plt.figure(figsize=(12, 6))
        for n_clusters, scores in all_results.items():
            yvals = [scores.get(m, 0.0) for m in all_modules]
            plt.plot(all_modules, yvals, marker='o', label=f"{n_clusters} clusters")

        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Relative Coverage (AvgCov / n_clusters)")
        plt.xlabel("Module")
        plt.title(f"Relative Coverage Across Cluster Configurations (≥ {threshold})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_path:
            if filename is None:
                filename = f'coverage_plot_threshold_{threshold:.2f}.png'

            save_path = Path(save_path)
            path = save_path / filename
            plt.savefig(path, dpi=300)
            print(f"Saved plot to {path}")
        else:
            plt.show()

    # Print best cluster count per module
    print("\nBEST NUMBER OF CLUSTERS PER MODULE")
    for module in all_modules:
        best_n = None
        best_val = -float("inf")
        for n_clusters, scores in all_results.items():
            v = scores.get(module, -float("inf"))
            if v > best_val:
                best_val = v
                best_n = n_clusters

        print(f"{module}: best = {best_n} clusters (relative coverage = {best_val:.6f})")
    return all_results





def empp_coverage_scores(**kwargs):
    """
    Computes and (optionally) plots class and cluster coverage per module

    Returns:
        dict: scores[module] = average_coverage
    """
    drillers = kwargs['drillers'] # dict drillers[peep_layer] = tGMM
    threshold = kwargs.get('threshold', 0.9) # threshold for coverage
    plot = kwargs.get('plot', False) # if True, plots the coverage
    #default_dir = next(iter(drillers.values()))._clas_path.parent
    default_dir = None
    save_path = kwargs.get('save_path', default_dir) # if plot is True
    filename = kwargs.get('file_name', None) 

    # Compute coverage
    class_scores,  cluster_scores = _get_empp_coverage_scores(drillers=drillers, threshold=threshold)

    scores = {}
    for module in class_scores:
        if module in cluster_scores:
            avg = (class_scores[module] + cluster_scores[module]) / 2
            scores[module] = avg
            print(f"[{module}] Avg Coverage: {avg:.3f}% (Class: {class_scores[module]:.3f}%, Cluster: {cluster_scores[module]:.3f}%)")

    # Optional plot
    if plot:
        modules = list(class_scores.keys())
        class_values = [class_scores[m] for m in modules]
        cluster_values = [cluster_scores.get(m, 0) for m in modules]

        plt.figure(figsize=(12, 6))
        plt.plot(modules, class_values, label='Class Coverage', marker='o')
        plt.plot(modules, cluster_values, label='Cluster Coverage', marker='x')

        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Coverage')
        plt.xlabel('Module')
        plt.title(f'Empirical Posterior Coverage (Threshold ≥ {threshold})')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_path:
            if filename is None:
                filename = f'coverage_plot_threshold_{threshold:.2f}.png'

            save_path = Path(save_path)
            path = save_path / filename
            plt.savefig(path, dpi=300)
            print(f"Saved plot to {path}")
        else:
            plt.show()
    return scores
