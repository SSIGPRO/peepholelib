import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import entropy
import torch
import math


def empirical_posterior_heatmaps(drillers, dir):
    os.makedirs(dir, exist_ok=True)

    for module_name, driller in drillers.items():
        driller.load()

        if not hasattr(driller, "_empp") or driller._empp is None:
            print(f"No empirical posterior found for {module_name}, skipping.")
            continue

        empp = driller._empp  # shape: [n_clusters, n_classes]
        print(f"Empirical Posterior shape for {module_name}: {empp.shape}")

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(empp.cpu().numpy(), aspect='auto', cmap='viridis', interpolation='nearest')

        ax.set_xlabel("Model Classes")
        ax.set_ylabel("Cluster Index")
        ax.set_title(f"Empirical Posterior - {module_name}")

        ax.set_xticks(np.linspace(0, empp.shape[1] - 1, min(20, empp.shape[1]), dtype=int))
        ax.set_yticks(np.linspace(0, empp.shape[0] - 1, min(20, empp.shape[0]), dtype=int))

        fig.colorbar(im, ax=ax, label="P(g|c)")

        filename = f"empirical_posterior_{module_name.replace('.', '_')}.png"
        filepath = os.path.join(dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved: {filepath}")

def get_empp_class_coverage_scores(drillers, threshold=0.9):
    """
    For each layer, check if each class is represented in at least a cluster,
    i.e., if any cluster assigns P(g|c) >= threshold for that class.

    Returns a dictionary [module - coverage_percent] 
    """
    scores = {}

    for module, driller in drillers.items():
        driller.load()

        if not hasattr(driller, "_empp") or driller._empp is None:
            print(f" No empirical posterior found for {module}, skipping.")
            continue

        empp = driller._empp  # [n_clusters, n_classes]
        empp = empp.cpu() if empp.is_cuda else empp

        empp_t = empp.T

        # For each class, check if any cluster has prob >= threshold
        class_represented = (empp_t >= threshold).any(dim=1)  # shape: [n_classes]
        n_represented = class_represented.sum().item()
        n_classes = empp.shape[1]
        coverage = 100 * n_represented / n_classes

        scores[module] = coverage

        print(f"[{module}] {n_represented}/{n_classes} classes represented (≥ {threshold}) — {coverage:.2f}%")

    return scores

def get_empp_cluster_coverage_scores(drillers, threshold=0.9):
    """
    For each layer, check if each cluster is associated with at least one class,
    i.e., if any class has P(g|c) >= threshold for that cluster.

    Returns a dictionary [module - coverage_percent]
    """
    scores = {}

    for module, driller in drillers.items():
        driller.load()

        if not hasattr(driller, "_empp") or driller._empp is None:
            print(f" No empirical posterior found for {module}, skipping.")
            continue

        empp = driller._empp  # [n_clusters, n_classes]
        empp = empp.cpu() if empp.is_cuda else empp

        # For each cluster, check if any class gets high enough probability
        cluster_active = (empp >= threshold).any(dim=1)  # shape: [n_clusters]
        n_active = cluster_active.sum().item()
        n_clusters = empp.shape[0]
        coverage = 100 * n_active / n_clusters

        scores[module] = coverage

        print(f"[{module}] {n_active}/{n_clusters} clusters represented(≥ {threshold}) — {coverage:.2f}%")

    return scores

def get_empp_coverage_scores(drillers, threshold=0.9):
    """
    Computes the average of class and cluster coverage for each layer.

    Args:
        drillers (dict): Dictionary of drillers per module.
        threshold (float): Threshold for considering a class or cluster represented.

    Returns:
        dict: scores[module] = average_coverage
    """
    class_scores = get_empp_class_coverage_scores(drillers, threshold)
    cluster_scores = get_empp_cluster_coverage_scores(drillers, threshold)

    scores = {}
    for module in class_scores:
        if module in cluster_scores:
            avg = (class_scores[module] + cluster_scores[module]) / 2
            scores[module] = avg
            print(f"[{module}] Avg Coverage: {avg:.2f}% (Class: {class_scores[module]:.2f}%, Cluster: {cluster_scores[module]:.2f}%)")

    return scores

def plot_empp_coverage_scores(drillers, threshold=0.9, save_path=None):
    """
    Plots class and cluster coverage across modules.

    Args:
        drillers (dict): Dict of drillers per module.
        threshold (float): Threshold for representation.
        save_path (str or Path, optional): If set, saves the figure to this path.
    """
    # Get both scores
    class_scores = get_empp_class_coverage_scores(drillers, threshold)
    cluster_scores = get_empp_cluster_coverage_scores(drillers, threshold)

    modules = list(class_scores.keys())
    class_values = [class_scores[m] for m in modules]
    cluster_values = [cluster_scores.get(m, 0) for m in modules]

    # Plot
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
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


