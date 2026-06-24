from math import ceil, floor, log, pi
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import adjusted_mutual_info_score

from .projection_optimization import _coverage, _fit_gmm,_get_labels_from_dataset,_gmm_num_parameters, _silhouette_score

def gride_comparison(**kwargs):
    """
    Compute GRIDE statistics on the projected corevectors and return them.
    """

    h_data = kwargs["h_data"]
    reduct_m = kwargs["reduct_m"]
    initial_cv_dim = int(kwargs["cv_dim"])
    verbose = kwargs.get("verbose", False)
    twonn_fraction = float(kwargs.get("twonn_fraction", 0.95))
    plot_path = kwargs.get("plot_path")
    double_large_discovered_cv_dim = bool(kwargs.get("double_large_discovered_cv_dim", True))

    if initial_cv_dim < 1:
        raise ValueError("cv_dim must be at least 1.")
    if not (0.0 < twonn_fraction <= 1.0):
        raise ValueError("twonn_fraction must be in the interval (0, 1].")
    if h_data.shape[1] != reduct_m.shape[1]:
        raise RuntimeError(
            f"Input dim mismatch: h_data has {h_data.shape[1]} features, "
            f"but reduct_m expects {reduct_m.shape[1]}."
        )

    device = reduct_m.device
    dtype = reduct_m.dtype
    full_reduct_m = reduct_m.detach().to(device=device, dtype=dtype)
    if initial_cv_dim > full_reduct_m.shape[0]:
        raise RuntimeError(f"cv_dim={initial_cv_dim} exceeds proj rank {full_reduct_m.shape[0]}.")

    h_data = h_data.detach().to(device=device, dtype=dtype)
    before_projected = _project_with_cv_dim(
        h_data=h_data,
        reduct_m=full_reduct_m,
        cv_dim=initial_cv_dim,
    )
    gride_n1 = int(kwargs.get("gride_n1", 10))
    gride_n2 = int(kwargs.get("gride_n2", 20))
    twonn_stats = gride_dimension(
        before_projected,
        fraction=twonn_fraction,
        n1=gride_n1,
        n2=gride_n2,
        return_diagnostics=True,
        verbose=verbose,
    )
    raw_discovered_cv_dim = int(max(1, ceil(float(twonn_stats["dimension"]))))
    if double_large_discovered_cv_dim and raw_discovered_cv_dim >= 20:
        raw_discovered_cv_dim *= 2
    discovered_cv_dim = int(max(1, min(full_reduct_m.shape[0], raw_discovered_cv_dim)))
    twonn_stats["cv_dim"] = discovered_cv_dim
    twonn_stats["initial_cv_dim"] = initial_cv_dim
    twonn_stats["best_cv_dim"] = discovered_cv_dim
    twonn_stats["optimized_cv_dim"] = discovered_cv_dim
    twonn_stats["cv_dim_candidates"] = [int(initial_cv_dim), discovered_cv_dim]
    twonn_stats["twonn_dimension"] = float(twonn_stats["dimension"])
    twonn_stats["twonn_cv_dim"] = discovered_cv_dim
    twonn_stats["gride_n1"] = int(gride_n1)
    twonn_stats["gride_n2"] = int(gride_n2)
    twonn_stats["plot_path"] = None if plot_path is None else Path(plot_path)

    if verbose:
        print(
            f"GRIDE estimate={twonn_stats['dimension']:.6f} "
            f"(n1={gride_n1}, n2={gride_n2}) "
            f"-> cv_dim={discovered_cv_dim} (original cv_dim={initial_cv_dim})"
        )

    requested_n_components = int(kwargs["n_components"])
    seed = kwargs.get("seed", 29)
    layer_name = kwargs.get("layer_name", kwargs.get("layer"))
    datasets = kwargs.get("datasets")
    loader = kwargs.get("loader", "train")
    label_key = kwargs.get("label_key", "label")
    coverage_threshold = float(kwargs.get("coverage_threshold", 0.8))
    n_classes = kwargs.get("n_classes")
    cluster_population_top_k = int(kwargs.get("cluster_population_top_k", 10))
    if seed is not None:
        torch.manual_seed(seed)
    if datasets is not None:
        labels = _get_labels_from_dataset(
            datasets=datasets,
            loader=loader,
            label_key=label_key,
            device=device,
        )
        if labels.shape[0] != h_data.shape[0]:
            raise ValueError(
                "h_data and datasets loader must contain the same number of samples. "
                f"Got {h_data.shape[0]} samples and {labels.shape[0]} labels."
            )
    else:
        labels = None
    after_projected = _project_with_cv_dim(
        h_data=h_data,
        reduct_m=full_reduct_m,
        cv_dim=discovered_cv_dim,
    )
    before_gmm = _fit_stable_gmm(
        projected=before_projected,
        n_components=requested_n_components,
        seed=seed,
        verbose=verbose,
    )
    after_gmm = _fit_stable_gmm(
        projected=after_projected,
        n_components=requested_n_components,
        seed=None if seed is None else seed + 1,
        verbose=verbose,
    )
    before_metrics = _compute_twonn_metrics(
        projected=before_projected,
        gmm_state=before_gmm,
        cv_dim=initial_cv_dim,
        requested_n_components=requested_n_components,
        labels=labels,
        coverage_threshold=coverage_threshold,
        n_classes=n_classes,
    )
    after_metrics = _compute_twonn_metrics(
        projected=after_projected,
        gmm_state=after_gmm,
        cv_dim=discovered_cv_dim,
        requested_n_components=requested_n_components,
        labels=labels,
        coverage_threshold=coverage_threshold,
        n_classes=n_classes,
    )
    twonn_stats["before_metrics"] = before_metrics
    twonn_stats["after_metrics"] = after_metrics
    if verbose:
        print(
            f"before: active={before_metrics['active_clusters']}/{requested_n_components}"
        )
        print(
            f"after:  active={after_metrics['active_clusters']}/{requested_n_components}"
        )
    plot_path_obj = None if plot_path is None else Path(plot_path)
    if plot_path_obj is not None:
        plot_path_obj.parent.mkdir(parents=True, exist_ok=True)
        _save_twonn_plot(
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            before_gmm=before_gmm,
            after_gmm=after_gmm,
            twonn_stats=twonn_stats,
            plot_path=plot_path_obj,
            layer_name=layer_name,
            requested_n_components=requested_n_components,
            cluster_population_top_k=cluster_population_top_k,
        )
    history = {
        "stage": ["before", "after_gride"],
        "cv_dim": [initial_cv_dim, discovered_cv_dim],
        "requested_n_components": [requested_n_components, requested_n_components],
        "active_clusters": [
            before_metrics["active_clusters"],
            after_metrics["active_clusters"],
        ],
        "twonn": twonn_stats,
    }
    return {
        "optimized_reduct_m": full_reduct_m.detach().clone(),
        "optimized_projection": full_reduct_m.detach().clone(),
        "optimized_cv_dim": discovered_cv_dim,
        "best_cv_dim": discovered_cv_dim,
        "initial_cv_dim": initial_cv_dim,
        "cv_dim_candidates": [initial_cv_dim, discovered_cv_dim],
        "before_projected": before_projected.detach().clone(),
        "after_projected": after_projected.detach().clone(),
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "before_gmm": _snapshot_gmm_state(before_gmm),
        "after_gmm": _snapshot_gmm_state(after_gmm),
        "history": history,
        "plot_path": plot_path_obj,
        "n_components": requested_n_components,
        "twonn": twonn_stats,
        "twonn_dimension": float(twonn_stats["dimension"]),
        "twonn_cv_dim": discovered_cv_dim,
}

def gride_dimension(data, fraction=0.9, n1=1, n2=2, return_diagnostics=False, verbose=False):
    """
    Estimate intrinsic dimension with the GRIDE likelihood from Denti et al.
    fraction is the fraction of valid points kept for the fit.
    """

    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError("fraction must be in the interval (0, 1].")

    if n2 <= n1:
        raise ValueError("n2 must be greater than n1.")

    distances = torch.cdist(data, data, p=2) # euclidean distance matrix
    nearest_distances = torch.topk(distances, k=n2 + 1, dim=1, largest=False).values
    k1 = nearest_distances[:, n1]
    k2 = nearest_distances[:, n2]

    eps = torch.finfo(nearest_distances.dtype).eps
    zero_mask = k1 <= eps
    degenerate_mask = torch.isclose(k1, k2, atol=eps, rtol=0.0)
    good_mask = ~(zero_mask | degenerate_mask)

    n_good = int(good_mask.sum().item())
    if verbose:
        print(f"Found {int(zero_mask.sum().item())} elements for which r1 = 0")
        print(f"Found {int(degenerate_mask.sum().item())} elements for which r_n1 = r_n2")
        print(f"Fraction good points: {n_good / max(1, data.shape[0]):.6f}")
    if n_good < 3:
        raise ValueError("Not enough non-degenerate samples for GRIDE.")

    mu = torch.sort(k2[good_mask] / k1[good_mask]).values
    npoints = min(max(1, int(floor(n_good * float(fraction)))), mu.numel())
    fit_mu = mu[:npoints]
    log_mu = torch.log(fit_mu.to(dtype=torch.float64))
    dimension = _solve_gride_mle(log_mu=log_mu, n1=n1, n2=n2)
    if n2 == n1 + 1:
        mg_dimension = _mg_dimension_from_log_mu(log_mu=log_mu, neighbor_order=n2)
    else:
        mg_dimension = None

    stats = {
        "dimension": float(dimension),
        "fraction": float(fraction),
        "n_samples": int(data.shape[0]),
        "n_good_points": n_good,
        "n_regression_points": int(npoints),
        "ignored_zero_first_neighbor": int(zero_mask.sum().item()),
        "ignored_equal_neighbors": int(degenerate_mask.sum().item()),
        "gride_n1": int(n1),
        "gride_n2": int(n2),
        "mg_dimension": None if mg_dimension is None else float(mg_dimension),
        "x": log_mu.to(dtype=mu.dtype).detach().cpu(),
        "y": _gride_cdf_transform(fit_mu, dimension=dimension, n1=n1, n2=n2).to(dtype=mu.dtype).detach().cpu(),
    }
    if verbose:
        print(
            f"for layer with shape {data.shape}, GRIDE estimated dimension: "
            f"{stats['dimension']:.6f} (n1={n1}, n2={n2})"
        )
        print (f"Used {stats['n_regression_points']} points for the fit, which is {stats['fraction']:.2%} of the {stats['n_good_points']} good points (out of {stats['n_samples']} total samples).")
        print(
            f"Ignored {stats['ignored_zero_first_neighbor']} points with zero lower-order "
            f"neighbor distance and {stats['ignored_equal_neighbors']} points with equal "
            f"n1/n2 neighbor distances."
        )
    if return_diagnostics:
        return stats
    return stats["dimension"]

def _solve_gride_mle(log_mu, n1, n2, min_dimension=1.0, max_iterations=80):
    '''
    binary search 
    '''
    low = float(min_dimension)
    low_score = _gride_score(low, log_mu=log_mu, n1=n1, n2=n2)
    if low_score <= 0.0:
        return low

    high = 2.0
    high_score = _gride_score(high, log_mu=log_mu, n1=n1, n2=n2)
    while high_score > 0.0 and high < 1e6:
        # double high until it returns negative
        high *= 2.0 
        high_score = _gride_score(high, log_mu=log_mu, n1=n1, n2=n2) 

    if high_score > 0.0:
        return high

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        mid_score = _gride_score(mid, log_mu=log_mu, n1=n1, n2=n2)
        if mid_score > 0.0:
            # is still increasing so the maximum is in the right
            low = mid
        else:
            # is decreasing so the maximum is in the left
            high = mid

    return 0.5 * (low + high)


def _gride_score(dimension, log_mu, n1, n2):
    '''
    calculates the "slope" for the given dimension
    score(d) = ∂logL(d) / ∂d
    L(d) = product of porbabilities resulted from the poisson process
    '''
    d = torch.tensor(float(dimension), device=log_mu.device, dtype=log_mu.dtype)
    z = (d * log_mu).clamp_min(torch.finfo(log_mu.dtype).eps) # d * log(mu)
    denom = (-torch.expm1(-z)).clamp_min(torch.finfo(log_mu.dtype).eps) # 1 - e^(-z)
    ratio_term = torch.sum(log_mu / denom) 
    # derivada de logL(d)
    score = (
        log_mu.numel() / d
        + (n2 - n1 - 1) * ratio_term
        - (n2 - 1) * torch.sum(log_mu)
    )
    return float(score.detach().cpu())


def _mg_dimension_from_log_mu(log_mu, neighbor_order):
    denominator = (neighbor_order - 1) * torch.sum(log_mu).clamp_min(1e-12)
    numerator = max(1, log_mu.numel() - 1)
    return float(numerator / denominator.detach().cpu())


def _gride_cdf_transform(mu, dimension, n1, n2):
    mu = mu.to(dtype=torch.float64)
    d = max(float(dimension), 1e-12)
    numerator = torch.pow(mu, d) - 1.0
    denominator = torch.pow(mu, (n2 - 1) * d)
    beta_pdf_factor = torch.pow(numerator.clamp_min(1e-12), n2 - n1 - 1)
    surrogate = beta_pdf_factor / denominator.clamp_min(1e-12)
    return torch.log1p(surrogate)


def _fit_stable_gmm(projected, n_components, seed=None, verbose=False, max_tries=20):
    last_evaluated_gmm = None

    for attempt_idx in range(max_tries):
        attempt_seed = None if seed is None else seed + attempt_idx
        raw_gmm = _fit_gmm(
            projected,
            n_components=n_components,
            seed=attempt_seed,
            cluster_method="gmm",
        )
        evaluated_gmm = _evaluate_gmm_state(projected, raw_gmm)
        last_evaluated_gmm = evaluated_gmm

        active_clusters = int((evaluated_gmm["cluster_counts"] > 0).sum().item())
        if active_clusters > 1 or int(n_components) <= 1:
            return evaluated_gmm

        if verbose:
            print(
                f"GMM fit attempt {attempt_idx + 1}/{max_tries} returned "
                f"{active_clusters} active cluster; retrying."
            )

    return last_evaluated_gmm


def _evaluate_gmm_state(data, gmm_state):
    weights = torch.as_tensor(gmm_state["weights"], device=data.device, dtype=data.dtype)
    means = torch.as_tensor(gmm_state["means"], device=data.device, dtype=data.dtype)
    variances = torch.as_tensor(gmm_state["variances"], device=data.device, dtype=data.dtype)

    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if float(weights.sum().detach().cpu()) <= 0.0:
        weights = torch.ones_like(weights) / max(1, weights.numel())
    else:
        weights = weights / weights.sum()
    means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
    variances = torch.nan_to_num(variances, nan=1e-6, posinf=1e6, neginf=1e-6).clamp_min(1e-6)

    log_prob = _gmm_component_log_prob(
        data=data,
        weights=weights,
        means=means,
        variances=variances,
    )
    log_norm = torch.logsumexp(log_prob, dim=1)
    posterior = torch.exp(log_prob - log_norm.unsqueeze(1))
    assignments = log_prob.argmax(dim=1)
    cluster_counts = torch.bincount(assignments, minlength=weights.shape[0]).to(device=data.device)

    return {
        "weights": weights.detach().clone(),
        "means": means.detach().clone(),
        "variances": variances.detach().clone(),
        "assignments": assignments.detach().clone(),
        "cluster_counts": cluster_counts.detach().clone(),
        "cluster_marginal_profile": posterior.mean(dim=0).detach().clone(),
        "max_assignment_probabilities": posterior.max(dim=1).values.detach().clone(),
        "nll": (-log_norm.sum()).detach().clone(),
    }


def _compute_twonn_metrics(projected, gmm_state, cv_dim,
                        requested_n_components,labels=None,
                        coverage_threshold=0.8,n_classes=None):
    counts = gmm_state["cluster_counts"]
    active_clusters = int((counts > 0).sum().item())

    n_samples = max(1, projected.shape[0])
    nll = _finite_or_nan(float(gmm_state["nll"].detach().cpu()))
    complexity = _gmm_num_parameters(
        n_components=requested_n_components,
        n_features=int(projected.shape[1]),
    )
    bic_penalty = float(complexity * log(max(2, n_samples)))
    bic = _finite_or_nan(2.0 * nll + bic_penalty)
    silhouette = _finite_or_nan(
        float(_silhouette_score(projected, gmm_state["assignments"].long()).detach().cpu())
    )

    metrics = {
        "cv_dim": int(cv_dim),
        "requested_n_components": int(requested_n_components),
        #"nll": nll,
        #"bic": bic,
        #"complexity": int(complexity),
        #"bic_penalty": bic_penalty,
        "active_clusters": int(active_clusters),
        "silhouette": silhouette,
        "md_col_mean": _mahalanobis_col_mean(gmm_state),
        "cluster_size_imbalance_ratio": _cluster_size_imbalance_ratio(counts),
    }

    if labels is not None:
        ami = _finite_or_nan(
            adjusted_mutual_info_score(
                labels.detach().cpu().numpy(),
                gmm_state["assignments"].long().detach().cpu().numpy(),
            )
        )
        metrics.update(
            {
                "ami": ami,
                **_coverage(
                assignments=gmm_state["assignments"].long(),
                labels=labels,
                n_clusters=gmm_state["weights"].shape[0],
                coverage_threshold=coverage_threshold,
                n_classes=n_classes,
                dtype=projected.dtype,
                ),
            }
        )

    return metrics


def _save_twonn_plot(before_metrics,after_metrics,
                    before_gmm,after_gmm,
                    twonn_stats,
                    plot_path,
                    layer_name,
                    requested_n_components,
                    cluster_population_top_k):
    
    fig = plt.figure(figsize=(18, 16))
    grid = fig.add_gridspec(3,2,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.55, 1.0, 1.35],
        hspace=0.45, wspace=0.25
    )

    ax_summary = fig.add_subplot(grid[0, :])
    ax_profile = fig.add_subplot(grid[1, :])
    ax_before_pop = fig.add_subplot(grid[2, 0])
    ax_after_pop = fig.add_subplot(grid[2, 1])

    fig.suptitle(f"Layer: {layer_name}", fontsize=16)

    _render_twonn_summary_panel(
        ax=ax_summary,
        twonn_stats=twonn_stats,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        requested_n_components=requested_n_components,
    )
    _render_cluster_marginal_profile(ax=ax_profile, before_gmm=before_gmm, after_gmm=after_gmm)
    _render_cluster_population_panel(
        ax=ax_before_pop,
        title="Cluster populations: original projection",
        gmm_state=before_gmm,
        cv_dim=before_metrics["cv_dim"],
        requested_n_components=requested_n_components,
        top_k=cluster_population_top_k,
    )
    _render_cluster_population_panel(
        ax=ax_after_pop,
        title="Cluster populations: GRIDE projection",
        gmm_state=after_gmm,
        cv_dim=after_metrics["cv_dim"],
        requested_n_components=requested_n_components,
        top_k=cluster_population_top_k,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_twonn_summary_panel(ax, twonn_stats, before_metrics, after_metrics, requested_n_components):
    ax.axis("off")
    summary_lines = [
        f"Original cv_dim: {twonn_stats['initial_cv_dim']}",
        f"GRIDE raw estimate: {twonn_stats['dimension']:.3f}",
        f"Neighbor orders: n1={twonn_stats['gride_n1']}, n2={twonn_stats['gride_n2']}",
        f"Used GRIDE cv_dim: {twonn_stats['cv_dim']}",
        f"Fixed GMM n_components: {requested_n_components}",
        (
            f"BIC before/after: "
            f"{_format_metric_value(before_metrics.get('bic'))} -> "
            f"{_format_metric_value(after_metrics.get('bic'))}"
        ),
        (
            f"NLL before/after: "
            f"{_format_metric_value(before_metrics.get('nll'))} -> "
            f"{_format_metric_value(after_metrics.get('nll'))}"
        ),
        (
            f"Silhouette before/after: "
            f"{_format_metric_value(before_metrics.get('silhouette'))} -> "
            f"{_format_metric_value(after_metrics.get('silhouette'))}"
        ),
        (
            f"MD col mean before/after: "
            f"{_format_metric_value(before_metrics.get('md_col_mean'))} -> "
            f"{_format_metric_value(after_metrics.get('md_col_mean'))}"
        ),
        (
            f"Cluster size imbalance before/after: "
            f"{_format_metric_value(before_metrics.get('cluster_size_imbalance_ratio'))} -> "
            f"{_format_metric_value(after_metrics.get('cluster_size_imbalance_ratio'))}"
        ),
        (
            f"Active clusters before/after: "
            f"{before_metrics['active_clusters']}/{requested_n_components} -> "
            f"{after_metrics['active_clusters']}/{requested_n_components}"
        ),
        (
            f"AMI before/after: "
            f"{_format_metric_value(before_metrics.get('ami'))} -> "
            f"{_format_metric_value(after_metrics.get('ami'))}"
        ) if "ami" in before_metrics or "ami" in after_metrics else None,
        (
            f"Class coverage before/after: "
            f"{_format_metric_value(before_metrics.get('class_coverage'))} -> "
            f"{_format_metric_value(after_metrics.get('class_coverage'))}"
        ) if "class_coverage" in before_metrics or "class_coverage" in after_metrics else None,
        (
            f"Cluster coverage before/after: "
            f"{_format_metric_value(before_metrics.get('cluster_coverage'))} -> "
            f"{_format_metric_value(after_metrics.get('cluster_coverage'))}"
        ) if "cluster_coverage" in before_metrics or "cluster_coverage" in after_metrics else None,
        f"Fit fraction kept: {twonn_stats['fraction']:.2f}",
        (
            f"Good points: {twonn_stats['n_good_points']}/{twonn_stats['n_samples']} | "
            f"Regression points: {twonn_stats['n_regression_points']}"
        ),
        (
            f"Ignored r_n1 = 0: {twonn_stats['ignored_zero_first_neighbor']} | "
            f"Ignored r_n1 = r_n2: {twonn_stats['ignored_equal_neighbors']}"
        ),
    ]
    ax.text(0.01,0.97,
        "\n".join(line for line in summary_lines if line is not None),
        va="top",ha="left",
        family="monospace",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7", edgecolor="#d0d0d0"),
    )
    ax.set_title("GRIDE summary", loc="left", fontsize=13)


def _render_cluster_marginal_profile(ax, before_gmm, after_gmm):
    before_profile = _sorted_cluster_profile(before_gmm)
    after_profile = _sorted_cluster_profile(after_gmm)

    if before_profile:
        ax.plot(
            list(range(1, len(before_profile) + 1)),
            before_profile,
            label="Before",
            color="tab:blue",
            linewidth=2.0,
            marker="o",
            markersize=3.0,
        )
    if after_profile:
        ax.plot(
            list(range(1, len(after_profile) + 1)),
            after_profile,
            label="After",
            color="tab:orange",
            linewidth=2.0,
            marker="o",
            markersize=3.0,
        )

    max_clusters = max(len(before_profile), len(after_profile))
    if max_clusters <= 1:
        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([1])
    else:
        ax.set_xlim(1, max_clusters)
        if max_clusters <= 20:
            ax.set_xticks(list(range(1, max_clusters + 1)))

    ax.set_title("Cluster posterior profile before/after GRIDE selection")
    ax.set_xlabel("Components ordered by descending posterior mass")
    ax.set_ylabel("Average posterior mass per component")
    ax.grid(True, axis="both", alpha=0.25)
    ax.legend()


def _render_cluster_population_panel(ax, title, gmm_state, cv_dim, requested_n_components, top_k):
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.01, 0.99,
        _cluster_population_summary(
            gmm_state=gmm_state,
            cv_dim=cv_dim,
            requested_n_components=requested_n_components,
            top_k=top_k,
        ),
        va="top", ha="left",
        family="monospace",
        fontsize=10
    )


def _cluster_population_summary(gmm_state, cv_dim, requested_n_components, top_k):
    counts = gmm_state["cluster_counts"].detach().cpu().tolist()
    indexed_counts = list(enumerate(counts))
    n_components = int(len(indexed_counts))
    active_clusters = sum(count > 0 for _, count in indexed_counts)
    total_samples = sum(counts)

    most_populated = sorted(indexed_counts, key=lambda item: (-item[1], item[0]))[:top_k]
    least_populated = sorted(indexed_counts, key=lambda item: (item[1], item[0]))[:top_k]

    lines = [
        f"cv_dim: {cv_dim}",
        f"Requested components: {requested_n_components}",
        f"Active clusters: {active_clusters}/{n_components}",
        f"Total samples: {total_samples}",
        "",
        f"Top {min(top_k, n_components)} most populated",
    ]
    lines.extend(_format_cluster_count_lines(most_populated))
    lines.extend([
        "",
        f"Top {min(top_k, n_components)} least populated",
    ])
    lines.extend(_format_cluster_count_lines(least_populated))
    return "\n".join(lines)


def _format_cluster_count_lines(indexed_counts):
    return [f"cluster {idx:>4}: {count}" for idx, count in indexed_counts]


def _sorted_cluster_profile(gmm_state):
    profile = torch.sort(
        torch.as_tensor(gmm_state["cluster_marginal_profile"]).detach().float().cpu(),
        descending=True,
    ).values
    return profile.tolist()


def _snapshot_gmm_state(gmm_state):
    return {
        "weights": gmm_state["weights"].detach().clone(),
        "means": gmm_state["means"].detach().clone(),
        "variances": gmm_state["variances"].detach().clone(),
        "assignments": gmm_state["assignments"].detach().clone(),
        "cluster_counts": gmm_state["cluster_counts"].detach().clone(),
        "cluster_marginal_profile": gmm_state["cluster_marginal_profile"].detach().clone(),
        "max_assignment_probabilities": gmm_state["max_assignment_probabilities"].detach().clone(),
        "nll": gmm_state["nll"].detach().clone(),
    }


def _project_with_cv_dim(h_data, reduct_m, cv_dim):
    return (reduct_m[:cv_dim] @ h_data.T).T


def _finite_or_nan(value):
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return float("nan")
    return value


def _mahalanobis_col_mean(gmm_state):
    matrix = _asymmetric_mahalanobis_matrix(gmm_state)
    if matrix is None or matrix.shape[0] <= 1:
        return 0.0

    eye = torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    col_mean = matrix.masked_fill(eye, 0.0).sum(dim=0) / (matrix.shape[0] - 1)
    return float(col_mean.mean().detach().cpu())


def _asymmetric_mahalanobis_matrix(gmm_state):
    if gmm_state is None:
        return None

    means = gmm_state.get("means")
    variances = gmm_state.get("variances")
    if means is None or variances is None:
        return None

    means = torch.as_tensor(means).detach().float().cpu()
    variances = torch.as_tensor(variances).detach().float().cpu().clamp_min(1e-12)

    inv_var = variances.reciprocal()
    weighted_means = means * inv_var
    t1 = means.pow(2) @ inv_var.T
    t2 = means @ weighted_means.T
    t3 = (means.pow(2) * inv_var).sum(dim=1)
    sq = t1 - 2.0 * t2 + t3.unsqueeze(0)
    sq.clamp_(min=0.0)
    return sq.sqrt()


def _cluster_size_imbalance_ratio(counts):
    counts = torch.as_tensor(counts).detach().float()
    positive_counts = counts[counts > 0]
    if positive_counts.numel() == 0:
        return float("nan")

    min_count = positive_counts.min()
    if float(min_count.detach().cpu()) <= 0.0:
        return float("nan")

    max_count = positive_counts.max()
    return float((max_count / min_count).detach().cpu())



def _format_metric_value(value):
    if value is None:
        return "-"
    value = float(value)
    if value != value:
        return "nan"
    return f"{value:.6f}"


def _gmm_component_log_prob(data, weights, means, variances):
    diff = data.unsqueeze(1) - means.unsqueeze(0)
    inv_var = variances.reciprocal()
    log_det = variances.log().sum(dim=1)
    mahalanobis = (diff.pow(2) * inv_var.unsqueeze(0)).sum(dim=2)
    n_features = data.shape[1]
    return (
        weights.clamp_min(1e-12).log().unsqueeze(0)
        - 0.5 * (n_features * log(2.0 * pi) + log_det.unsqueeze(0) + mahalanobis)
    )
