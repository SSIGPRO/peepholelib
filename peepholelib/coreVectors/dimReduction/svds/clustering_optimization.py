from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tensordict import PersistentTensorDict
from torchgmm.bayes import GaussianMixture as tGMM
from torchgmm.bayes.gmm.model import GaussianMixtureModel, GaussianMixtureModelConfig

from .projection_optimization import (_compute_clustering_metrics,_evaluate_gmm,_fit_gmm,_format,_get_labels_from_dataset,
                                         _compute_empp_from_class_ids,_labels_to_class_ids,
                                         _gmm_num_parameters,_metric_rows,_render_cluster_marginal_likelihood_panel,
                                         _render_cluster_population_panel,_safe_relative_improvement,
                                         _select_layer_value,_snapshot_gmm_state,_target_layers_from_kwargs,
                                         optimize_projection)

def optimize_clustering(**kwargs):
    """
    Two-stage clustering optimization pipeline.
    1. Freeze a fitted mixture model and optimize the projection weights for
       the input cv_dim.
    2. Refit clustering on the optimized projection.
    """
    target_layers = _target_layers_from_kwargs(kwargs)
    if target_layers is not None:
        results = {}
        default_driller_name = "driller_name" not in kwargs
        for layer in target_layers:
            layer_kwargs = dict(kwargs)
            layer_kwargs.pop("target_layers", None)
            layer_kwargs["layer_name"] = layer
            layer_kwargs["layer"] = layer
            layer_kwargs["h_data"] = _select_layer_value(
                layer_kwargs["h_data"], layer, "h_data", target_layers
            )
            layer_kwargs["reduct_m"] = _select_layer_value(
                layer_kwargs["reduct_m"], layer, "reduct_m", target_layers
            )
            layer_kwargs["cv_dim"] = _select_layer_value(
                layer_kwargs["cv_dim"], layer, "cv_dim", target_layers, shared_ok=True
            )
            layer_kwargs["n_components"] = _select_layer_value(
                layer_kwargs["n_components"], layer, "n_components", target_layers, shared_ok=True
            )
            if "plot_path" in layer_kwargs:
                layer_kwargs["plot_path"] = _select_layer_value(
                    layer_kwargs["plot_path"], layer, "plot_path", target_layers, shared_ok=True
                )
            if "projection_plot_path" in layer_kwargs:
                layer_kwargs["projection_plot_path"] = _select_layer_value(
                    layer_kwargs["projection_plot_path"], layer, "projection_plot_path", target_layers, shared_ok=True
                )
            if "driller_name" in layer_kwargs:
                layer_kwargs["driller_name"] = _select_layer_value(
                    layer_kwargs["driller_name"], layer, "driller_name", target_layers, shared_ok=True
                )
            if default_driller_name:
                layer_kwargs["driller_name"] = None
            results[layer] = optimize_clustering(**layer_kwargs)
        return results

    h_data = kwargs["h_data"]
    reduct_m = kwargs["reduct_m"]
    initial_cv_dim = int(kwargs["cv_dim"])
    initial_n_components = int(kwargs["n_components"])
    plot_path = None if kwargs.get("plot_path") is None else Path(kwargs["plot_path"])
    loss_name = kwargs.get("loss", "bic").lower()
    show_mahalanobis_matrices = bool(kwargs.get("show_mahalanobis_matrices", True))
    projection_loss = kwargs.get("projection_loss", loss_name).lower()
    seed = kwargs.get("seed", 29)
    verbose = kwargs.get("verbose", False)
    projection_plot_path = kwargs.get("projection_plot_path", None)
    layer_name = kwargs.get("layer_name", kwargs.get("layer"))
    datasets = kwargs.get("datasets")
    loader = kwargs.get("loader", "train")
    label_key = kwargs.get("label_key", "label")
    coverage_threshold = float(kwargs.get("coverage_threshold", 0.8))
    cluster_population_top_k = int(kwargs.get("cluster_population_top_k", 10))
    cluster_method = str(kwargs.get("cluster_method", "gmm")).strip().lower()
    dpgmm_max_clusters = int(kwargs.get("dpgmm_max_clusters", 100))
    dpgmm_iterations = int(kwargs.get("dpgmm_iterations", 1000))
    gmm_retries = int(kwargs.get("gmm_retries", 10))
    covariance_type = str(kwargs.get("covariance_type", "diag")).strip().lower()
    # saving parameters
    cv_path = kwargs.get("cv_path", None)
    cv_name = kwargs.get("cv_name", "optimized_corevectors")
    save_corevectors_loader = kwargs.get("save_corevectors_loader", kwargs.get("corevectors_loader", loader))
    drillers_path = kwargs.get("drillers_path", kwargs.get("drillers_path"))
    driller_name = kwargs.get("driller_name")

    labels = None
    if drillers_path is not None:
        labels = _get_labels_from_dataset(
            datasets=datasets,
            loader=loader,
            label_key=label_key,
            device=h_data.device,
        )

    projection_results = optimize_projection(
        h_data=h_data,
        reduct_m=reduct_m,
        cv_dim=initial_cv_dim,
        loss=projection_loss,
        n_components=initial_n_components,
        plot_path=projection_plot_path,
        seed=seed,
        verbose=verbose,
        datasets=datasets,
        loader=loader,
        label_key=label_key,
        layer_name=layer_name,
        coverage_threshold=coverage_threshold,
        n_classes=kwargs.get("n_classes"),
        cluster_population_top_k=cluster_population_top_k,
        cluster_method=cluster_method,
        dpgmm_max_clusters=dpgmm_max_clusters,
        dpgmm_iterations=dpgmm_iterations,
        gmm_retries=gmm_retries,
        covariance_type=covariance_type,
    )

    optimized_reduct_m = projection_results["optimized_reduct_m"].detach().clone()
    h_data = h_data.detach().to(device=optimized_reduct_m.device, dtype=optimized_reduct_m.dtype)
    optimized_projected_raw = projection_results["after_projected"].detach().to(
        device=optimized_reduct_m.device,
        dtype=optimized_reduct_m.dtype,
    )
    normalization_mean, normalization_std = _compute_projection_normalization(optimized_projected_raw)
    optimized_projected = (optimized_projected_raw - normalization_mean) / normalization_std
    before_gmm_state = _normalize_gmm_state(
        projection_results["after_gmm"],
        mean=normalization_mean,
        std=normalization_std,
        normalized_projected=optimized_projected,
    )
    before_component_count = int(before_gmm_state["weights"].shape[0])
    before_n_params = int(
        before_gmm_state.get(
            "n_params",
            _gmm_num_parameters(
                n_components=before_component_count,
                n_features=optimized_projected.shape[1],
            ),
        )
    )
    before_metrics = _compute_clustering_metrics(
        optimized_projected,
        before_gmm_state,
        "bic",
        before_n_params,
        h_data.shape[0],
        None if seed is None else seed + 49,
        labels,
        coverage_threshold,
        kwargs.get("n_classes"),
    )
    before_metrics.pop("objective", None)
    before_metrics = {"bic": float("nan")}

    after_seed = None if seed is None else seed + 42
    refit_n_components = (
        before_component_count
        if cluster_method == "gmm"
        else dpgmm_max_clusters
    )
    after_gmm_state = _fit_gmm(
        optimized_projected,
        n_components=refit_n_components,
        seed=after_seed,
        cluster_method=cluster_method,
        dpgmm_max_clusters=dpgmm_max_clusters,
        dpgmm_iterations=dpgmm_iterations,
        gmm_retries=gmm_retries,
        covariance_type=covariance_type,
    )
    optimized_n_components = int(after_gmm_state["weights"].shape[0])
    after_n_params = _gmm_num_parameters(
        n_components=optimized_n_components,
        n_features=optimized_projected.shape[1],
    )
    after_metrics = _compute_clustering_metrics(
        optimized_projected,
        after_gmm_state,
        "bic",
        after_n_params,
        h_data.shape[0],
        None if after_seed is None else after_seed + 49,
        labels,
        coverage_threshold,
        kwargs.get("n_classes"),
    )
    after_metrics.pop("objective", None)
    after_metrics = {"bic": float("nan")}

    history = {
        "stage": ["projection_optimized_state", "refit_active_clusters"],
        "cv_dim": [initial_cv_dim, initial_cv_dim],
        "n_components": [before_component_count, optimized_n_components],
        "nll": [before_metrics["nll"], after_metrics["nll"]],
        "bic": [before_metrics["bic"], after_metrics["bic"]],
        "silhouette": [before_metrics["silhouette"], after_metrics["silhouette"]],
        "normalized_cluster_entropy": [
            before_metrics["normalized_cluster_entropy"],
            after_metrics["normalized_cluster_entropy"],
        ],
        "active_clusters": [before_metrics["active_clusters"], after_metrics["active_clusters"]],
        "class_coverage": [
            before_metrics.get("class_coverage"),
            after_metrics.get("class_coverage"),
        ],
        "cluster_coverage": [
            before_metrics.get("cluster_coverage"),
            after_metrics.get("cluster_coverage"),
        ],
    }
    history = None

    if verbose:
        print(
            f"cv_dim={initial_cv_dim} cluster_method={cluster_method} "
            f"components={before_component_count} -> {optimized_n_components}"
        )
        print(
            f"projection_state bic={before_metrics['bic']:.6f} "
            f"nll={before_metrics['nll']:.6f} silhouette={before_metrics['silhouette']:.6f}"
        )
        print(
            f"refit_active_clusters bic={after_metrics['bic']:.6f} "
            f"nll={after_metrics['nll']:.6f} silhouette={after_metrics['silhouette']:.6f}"
        )

    before_gmm_snapshot = _snapshot_gmm_state(
        before_gmm_state,
        n_params=before_n_params,
        n_samples=h_data.shape[0],
    )
    after_gmm_snapshot = _snapshot_gmm_state(
        after_gmm_state,
        n_params=after_n_params,
        n_samples=h_data.shape[0],
    )

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        _save_cluster_count_clustering_stats_plot(
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            before_gmm=before_gmm_snapshot,
            after_gmm=after_gmm_snapshot,
            plot_path=plot_path,
            layer_name=layer_name,
            cv_dim=initial_cv_dim,
            initial_n_components=initial_n_components,
            optimized_n_components=optimized_n_components,
            cluster_population_top_k=cluster_population_top_k,
            cluster_population_mode="both",
            cluster_method=cluster_method,
            show_mahalanobis_matrices=show_mahalanobis_matrices,
        )

    saved_cv_path = None
    if cv_path is not None:
        saved_cv_path = _save_optimized_corevectors(
            projected=optimized_projected,
            layer_name=layer_name,
            path=cv_path,
            name=cv_name,
            loader=save_corevectors_loader,
            verbose=verbose,
        )
        _save_corevector_normalization(
            mean=normalization_mean,
            std=normalization_std,
            layer_name=layer_name,
            path=cv_path,
            verbose=verbose,
        )

    saved_driller_path = None
    if drillers_path is not None:
        if driller_name is None:
            driller_name = (
                f"optimized_driller.{cluster_method.upper()}."
                f"{layer_name}.{initial_n_components}."
                f"{optimized_n_components}.{optimized_projected.shape[1]}"
            )
        saved_driller_path = _save_optimized_gmm_driller(
            gmm_state=after_gmm_state,
            labels=labels,
            path=drillers_path,
            name=driller_name,
            label_key=label_key,
            n_classes=kwargs.get("n_classes"),
            covariance_type=covariance_type,
            verbose=verbose,
        )

    return {
        "optimized_reduct_m": optimized_reduct_m,
        "optimized_projection": optimized_reduct_m.detach().clone(),
        "optimized_cv_dim": initial_cv_dim,
        "best_cv_dim": initial_cv_dim,
        "initial_cv_dim": initial_cv_dim,
        "cv_dim_candidates": [initial_cv_dim],
        "initial_n_components": initial_n_components,
        "optimized_n_components": optimized_n_components,
        "projection_active_clusters": optimized_n_components,
        "before_projected": optimized_projected.detach().clone(),
        "after_projected": optimized_projected.detach().clone(),
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "before_gmm": before_gmm_snapshot,
        "after_gmm": after_gmm_snapshot,
        "history": history,
        "plot_path": plot_path,
        "loss": loss_name,
        "projection_loss": projection_loss,
        "n_components": optimized_n_components,
        "cluster_method": cluster_method,
        "dpgmm_max_clusters": dpgmm_max_clusters,
        "gmm_retries": gmm_retries,
        "projection_optimization": projection_results,
        "coverage_threshold": coverage_threshold,
        "saved_cv_path": saved_cv_path,
        "saved_driller_path": saved_driller_path,
    }


def _save_cluster_count_clustering_stats_plot(**kwargs):
    before_metrics = kwargs["before_metrics"]
    after_metrics = kwargs["after_metrics"]
    before_gmm = kwargs.get("before_gmm")
    after_gmm = kwargs.get("after_gmm")
    plot_path = Path(kwargs["plot_path"])
    layer_name = kwargs.get("layer_name", "")
    cluster_population_top_k = int(kwargs.get("cluster_population_top_k", 10))
    show_mahalanobis_matrices = bool(kwargs.get("show_mahalanobis_matrices", True))
    cluster_method = kwargs.get("cluster_method", "gmm").upper()
    before_label = f"Frozen {cluster_method}"
    after_label = f"Refit {cluster_method}"
    summary_text = (
        f"Fixed cv_dim: {kwargs['cv_dim']} | "
        f"components: {kwargs['before_gmm']['weights'].shape[0]} -> "
        f"{kwargs['optimized_n_components']}"
    )

    rows = _metric_rows(before_metrics, after_metrics)
    table_data = []
    for key, label in rows:
        before = before_metrics[key]
        after = after_metrics[key]
        delta = None if before is None or after is None else after - before
        table_data.append([label, _format(before), _format(after), _format(delta)])

    fig = plt.figure(figsize=(18, 19 if show_mahalanobis_matrices else 14))
    grid = fig.add_gridspec(
        4 if show_mahalanobis_matrices else 3,
        2,
        width_ratios=[1.35, 1.0],
        height_ratios=(
            [1.0, 0.9, 1.1, 1.15]
            if show_mahalanobis_matrices
            else [1.0, 0.9, 1.1]
        ),
    )
    ax_summary = fig.add_subplot(grid[0, 0])
    ax_improvement = fig.add_subplot(grid[0, 1])
    ax_marginal = fig.add_subplot(grid[1, :])
    ax_population_before = fig.add_subplot(grid[2, 0])
    ax_population_after = fig.add_subplot(grid[2, 1])
    if show_mahalanobis_matrices:
        ax_matrix_before = fig.add_subplot(grid[3, 0])
        ax_matrix_after = fig.add_subplot(grid[3, 1])

    ax_summary.axis("off")
    table = ax_summary.table(
        cellText=table_data,
        colLabels=["Metric", "Before", "After", "Delta"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)
    ax_summary.set_title("Clustering refit quality summary")
    ax_summary.text(
        0.5, 0.04,
        summary_text,
        ha="center", va="bottom",
        transform=ax_summary.transAxes,
        fontsize=10,
    )

    fig.suptitle(f"Layer: {layer_name}", fontsize=14)

    improvement_rows = [
        ("NLL", _safe_relative_improvement(before_metrics["nll"], after_metrics["nll"], lower_is_better=True)),
        ("BIC", _safe_relative_improvement(before_metrics.get("bic"), after_metrics.get("bic"), lower_is_better=True)),
        ("Silhouette", _safe_relative_improvement(before_metrics["silhouette"], after_metrics["silhouette"], lower_is_better=False)),
        (
            "Norm. entropy",
            _safe_relative_improvement(
                before_metrics["normalized_cluster_entropy"],
                after_metrics["normalized_cluster_entropy"],
                lower_is_better=False,
            ),
        ),
        (
            "Class coverage",
            _safe_relative_improvement(
                before_metrics.get("class_coverage"),
                after_metrics.get("class_coverage"),
                lower_is_better=False,
            ),
        ),
        (
            "Cluster coverage",
            _safe_relative_improvement(
                before_metrics.get("cluster_coverage"),
                after_metrics.get("cluster_coverage"),
                lower_is_better=False,
            ),
        ),
    ]
    labels = [label for label, value in improvement_rows if value is not None]
    values = [value for _, value in improvement_rows if value is not None]

    ax_improvement.axvline(0.0, color="black", linewidth=1.0)
    colors = ["tab:green" if value >= 0 else "tab:red" for value in values]
    ax_improvement.barh(labels, values, color=colors)
    ax_improvement.set_xlabel("Relative improvement")
    ax_improvement.set_title(f"Clustering improvement after {cluster_method} refit")
    ax_improvement.set_xlim(
        min(-0.05, min(values + [0.0]) * 1.1 if values else -0.05),
        max(0.05, max(values + [0.0]) * 1.1 if values else 0.05),
    )
    for idx, value in enumerate(values):
        ax_improvement.text(value, idx, f" {value * 100:.2f}%", va="center")

    _render_cluster_marginal_likelihood_panel(
        ax=ax_marginal,
        title=f"Cluster marginal posterior profile after {cluster_method} refit",
        before_gmm=before_gmm,
        after_gmm=after_gmm,
    )
    ax_population_before.axis("off")
    ax_population_after.axis("off")
    _render_cluster_population_panel(
        ax=ax_population_before,
        title=f"Frozen {cluster_method} cluster populations",
        gmm_state=before_gmm,
        top_k=cluster_population_top_k,
    )
    _render_cluster_population_panel(
        ax=ax_population_after,
        title=f"Refit {cluster_method} cluster populations",
        gmm_state=after_gmm,
        top_k=cluster_population_top_k,
    )

    if show_mahalanobis_matrices:
        _render_mahalanobis_matrix_panels(
            before_ax=ax_matrix_before, after_ax=ax_matrix_after,
            before_gmm=before_gmm, after_gmm=after_gmm,
            before_label=before_label, after_label=after_label
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97) if layer_name else None)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_mahalanobis_matrix_panels(before_ax, after_ax, before_gmm, after_gmm, before_label, after_label):
    before_matrix = _asymmetric_mahalanobis_matrix(before_gmm)
    after_matrix = _asymmetric_mahalanobis_matrix(after_gmm)
    matrices = [matrix for matrix in (before_matrix, after_matrix) if matrix is not None]

    if not matrices:
        for ax in (before_ax, after_ax):
            ax.text(0.5, 0.5, "No GMM state available.", ha="center", va="center")
            ax.set_axis_off()
        return

    finite_maxima = [matrix.max() for matrix in matrices if bool(torch.isfinite(matrix).all().item())]
    vmax = torch.stack(finite_maxima).max() if finite_maxima else torch.tensor(1.0)
    vmax = float(vmax.detach().cpu())
    if vmax <= 0:
        vmax = 1.0

    for ax, matrix, label in (
        (before_ax, before_matrix, before_label),
        (after_ax, after_matrix, after_label),
    ):
        if matrix is None:
            ax.text(0.5, 0.5, "No GMM state available.", ha="center", va="center")
            ax.set_axis_off()
            continue

        im = ax.imshow(matrix.numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        mean_col_dist, max_dist = _mahalanobis_matrix_summary(matrix)
        ax.set_title(
            f"{label} centroid Mahalanobis matrix\n"
            f"K={matrix.shape[0]}  col_mean={mean_col_dist:.3f}  max={max_dist:.3f}",
            fontsize=9,
        )
        ax.set_xlabel(r"j  (reference $\Sigma_j$)", fontsize=8)
        ax.set_ylabel("i", fontsize=8)
        ax.tick_params(labelsize=7)


def _asymmetric_mahalanobis_matrix(gmm_state):
    if gmm_state is None:
        return None

    means = torch.as_tensor(gmm_state["means"]).detach().float().cpu()
    variances = torch.as_tensor(gmm_state["variances"]).detach().float().cpu()
    variances = variances.clamp_min(1e-12)

    inv_var = variances.reciprocal()
    weighted_means = means * inv_var
    t1 = means.pow(2) @ inv_var.T
    t2 = means @ weighted_means.T
    t3 = (means.pow(2) * inv_var).sum(dim=1)
    sq = t1 - 2.0 * t2 + t3.unsqueeze(0)
    sq.clamp_(min=0.0)
    return sq.sqrt()


def _mahalanobis_matrix_summary(matrix):
    max_dist = float(matrix.max().detach().cpu())
    if matrix.shape[0] <= 1:
        return 0.0, max_dist

    eye = torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    col_mean = matrix.masked_fill(eye, 0.0).sum(dim=0) / (matrix.shape[0] - 1)
    return float(col_mean.mean().detach().cpu()), max_dist


def _compute_projection_normalization(projected):
    mean = projected.mean(dim=0)
    std = projected.std(dim=0)
    return mean, std


def _normalize_gmm_state(gmm_state, mean, std, normalized_projected):
    device = normalized_projected.device
    dtype = normalized_projected.dtype
    mean = mean.to(device=device, dtype=dtype)
    std = std.to(device=device, dtype=dtype)

    normalized_state = dict(gmm_state)
    normalized_state["weights"] = gmm_state["weights"].detach().to(device=device, dtype=dtype)
    normalized_state["means"] = (
        gmm_state["means"].detach().to(device=device, dtype=dtype) - mean
    ) / std
    normalized_state["variances"] = (
        gmm_state["variances"].detach().to(device=device, dtype=dtype) / std.pow(2)
    )

    assignments = gmm_state.get("assignments")
    if assignments is not None:
        assignments = assignments.detach().to(device=device)
    return _evaluate_gmm(
        data=normalized_projected,
        gmm_state=normalized_state,
        assignments=assignments,
    )


def _save_optimized_corevectors(projected, layer_name, path, name, loader, verbose=False):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{name}.{loader}"
    projected = projected.detach().cpu()

    if file_path.exists():
        corevds = PersistentTensorDict.from_h5(file_path, mode="r+")
    else:
        corevds = PersistentTensorDict(filename=file_path, batch_size=[projected.shape[0]], mode="w")

    try:
        corevds[layer_name] = projected
        if verbose:
            print(f"Saved optimized corevectors to {file_path} [{layer_name}]")
    finally:
        corevds.close()

    return file_path


def _save_corevector_normalization(mean, std, layer_name, path, verbose=False):
    path = Path(path)
    norm_path = path / "optimized_corevectors.normalization.pt"
    norm_path.parent.mkdir(parents=True, exist_ok=True)

    if norm_path.exists():
        means, stds = torch.load(norm_path, weights_only=False)
    else:
        means, stds = {}, {}

    means[layer_name] = mean.detach().cpu()
    stds[layer_name] = std.detach().cpu()
    torch.save((means, stds), norm_path)

    if verbose:
        print(f"Saved optimized corevector normalization to {norm_path} [{layer_name}]")
    return norm_path


def _save_optimized_gmm_driller(gmm_state,labels,path,name,label_key,n_classes=None,covariance_type="diag",verbose=False):
    path = Path(path)
    clas_path = path / name

    path.mkdir(parents=True, exist_ok=True)
    clas_path.mkdir(parents=True, exist_ok=True)

    estimator = _gmm_estimator_from_state(gmm_state, covariance_type=covariance_type)
    estimator.save(clas_path)

    labels, n_classes = _labels_to_class_ids(
        labels=labels,
        n_classes=n_classes,
        device=gmm_state["assignments"].device,
    )
    empp = _compute_empp_from_class_ids(
        assignments=gmm_state["assignments"].long(),
        labels=labels,
        n_clusters=gmm_state["weights"].shape[0],
        n_classes=n_classes,
        dtype=gmm_state["means"].dtype,
    )
    torch.save(empp.detach().cpu(), clas_path / f"empp_{label_key}.pt")

    if verbose:
        print(f"Saved optimized driller to {clas_path}")
    return clas_path


def _gmm_estimator_from_state(gmm_state, covariance_type="diag"):
    weights = gmm_state["weights"].detach()
    means = gmm_state["means"].detach()
    variances = gmm_state["variances"].detach().clamp_min(1e-12)
    n_components, n_features = means.shape

    device = means.device
    trainer_params = {
        "num_nodes": 1,
        "max_epochs": 100,
        "accelerator": device.type,
        "devices": [device.index] if device.type == "cuda" else 1,
        "enable_progress_bar": False,
    }

    estimator = tGMM(
        num_components=int(n_components),
        covariance_type=covariance_type,
        trainer_params=trainer_params,
    )
    model = GaussianMixtureModel(
        GaussianMixtureModelConfig(
            num_components=int(n_components),
            num_features=int(n_features),
            covariance_type=covariance_type,
        )
    ).to(device=means.device, dtype=means.dtype)

    with torch.no_grad():
        model.component_probs.copy_(weights / weights.sum().clamp_min(1e-12))
        model.means.copy_(means)
        model.precisions_cholesky.copy_(variances.rsqrt())

    estimator.model_ = model
    estimator.converged_ = True
    estimator.num_iter_ = 0
    n_samples = gmm_state.get("cluster_counts")
    n_samples = int(n_samples.sum().item()) if n_samples is not None else int(gmm_state["assignments"].numel())
    estimator.nll_ = float((gmm_state["nll"] / max(1, n_samples)).detach().cpu())
    return estimator
