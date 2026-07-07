from math import ceil, exp, lgamma, log, pi
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import adjusted_mutual_info_score

from .projection_optimization import _coverage,_fit_gmm, _get_labels_from_dataset, _gmm_num_parameters, _silhouette_score


def hidalgo_comparison(**kwargs):
    """
    Run a Hidalgo-style local-ID segmentation on the projected corevectors and
    compare fixed-GMM clustering statistics before and after the discovered
    global corevector dimension.
    Args:
        h_data: The original high-dimensional data (n_samples, n_features)
        reduct_m: The projection matrix used to reduce the dimensionality of h_data.
        cv_dim: The initial corevector dimension used for projection.
        n_components: The number of components for the GMM clustering.
        plot_path (opt): If provided, save a plot summarizing the Hidalgo results to this path.
        fraction: Retained for API compatibility. Hidalgo uses all valid points for
        estimation, matching the paper.
        k (opt): Number of segments for Hidalgo. Default is 2.
        q (opt): Number of nearest neighbors for Hidalgo. Default is 3.
        zeta: Used for encoraging near neighbours to have the same ID. 
        If zeta = 0.5 there's no encoraging. Default is 0.8.
        n_restarts: Number of restarts for Hidalgo optimization. Default is 6.
        n_iter (opt): Max number of iterations for Hidalgo optimization. Default is 200.
        tol (opt): Tolerance for convergence in Hidalgo optimization. Default is 1e-5.
        confidence_threshold: Threshold for segment assignments in Hidalgo. Default is 0.8.
        min_segment_weight: Min weight for segments in Hidalgo. Default is 1e-3.
    """

    h_data = kwargs["h_data"]
    reduct_m = kwargs["reduct_m"]
    initial_cv_dim = int(kwargs["cv_dim"])
    requested_n_components = int(kwargs["n_components"])
    verbose = bool(kwargs.get("verbose", False))
    plot_path = kwargs.get("plot_path")

    fraction = float(kwargs.get("fraction", kwargs.get("twonn_fraction", 0.95)))
    k = int(kwargs.get("k", 2))
    q = int(kwargs.get("q", 3))
    zeta = float(kwargs.get("zeta", 0.8))
    n_restarts = int(kwargs.get("n_restarts", 6))
    n_iter = int(kwargs.get("n_iter", 100))
    tol = float(kwargs.get("tol", 1e-5))
    confidence_threshold = float(kwargs.get("confidence_threshold", 0.8))
    min_segment_weight = float(kwargs.get("min_segment_weight", 1e-3))
    if zeta < 0.5:
        raise ValueError(f"zeta < 0.5 doesn't make sense. Why are you discouraging neighbors from having the same ID?")
    if h_data.shape[1] != reduct_m.shape[1]:
        raise RuntimeError(
            f"Input dim mismatch: h_data has {h_data.shape[1]} features, "
            f"but reduct_m expects {reduct_m.shape[1]}."
        )

    device = reduct_m.device
    dtype = reduct_m.dtype
    full_reduct_m = reduct_m.to(device=device, dtype=dtype)
    if initial_cv_dim > full_reduct_m.shape[0]:
        raise RuntimeError(f"cv_dim={initial_cv_dim} exceeds proj rank {full_reduct_m.shape[0]}.")

    h_data = h_data.to(device=device, dtype=dtype)
    before_projected = _project_with_cv_dim(
        h_data=h_data,
        reduct_m=full_reduct_m,
        cv_dim=initial_cv_dim,
    )

    hidalgo_stats = hidalgo_dimension(
        before_projected,
        fraction=fraction,
        k=k, q=q, zeta=zeta,
        n_iter=n_iter,
        n_restarts=n_restarts,
        tol=tol,
        confidence_threshold=confidence_threshold,
        min_segment_weight=min_segment_weight,
        return_diagnostics=True,
        verbose=verbose,
        seed=kwargs.get("seed", 29),
    )
    raw_discovered_cv_dim = int(max(1, ceil(float(hidalgo_stats["dimension"]))))
    discovered_cv_dim = int(max(1, min(full_reduct_m.shape[0], raw_discovered_cv_dim)))

    hidalgo_stats["cv_dim"] = discovered_cv_dim
    hidalgo_stats["initial_cv_dim"] = initial_cv_dim
    hidalgo_stats["best_cv_dim"] = discovered_cv_dim
    hidalgo_stats["optimized_cv_dim"] = discovered_cv_dim
    hidalgo_stats["cv_dim_candidates"] = [int(initial_cv_dim), int(discovered_cv_dim)]
    hidalgo_stats["hidalgo_dimension"] = float(hidalgo_stats["dimension"])
    hidalgo_stats["hidalgo_cv_dim"] = int(discovered_cv_dim)
    hidalgo_stats["plot_path"] = None if plot_path is None else Path(plot_path)

    if verbose:
        print(
            f"Hidalgo effective dimension={hidalgo_stats['dimension']:.6f} "
            f"(K={k}, q={q}, zeta={zeta:.3f}) "
            f"-> cv_dim={discovered_cv_dim} (original cv_dim={initial_cv_dim})"
        )

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

    before_gmm = _fit_stable_gmm(
        projected=before_projected,
        n_components=requested_n_components,
        seed=seed,
        verbose=verbose,
    )
    before_metrics = _compute_hidalgo_metrics(
        projected=before_projected,
        gmm_state=before_gmm,
        cv_dim=initial_cv_dim,
        requested_n_components=requested_n_components,
        labels=labels,
        coverage_threshold=coverage_threshold,
        n_classes=n_classes,
    )
    after_state = _compute_hidalgo_max_dim_after_state(
        h_data=h_data,
        reduct_m=full_reduct_m,
        requested_n_components=requested_n_components,
        hidalgo_stats=hidalgo_stats,
        labels=labels,
        coverage_threshold=coverage_threshold,
        n_classes=n_classes,
        full_rank=int(full_reduct_m.shape[0]),
        seed=None if seed is None else seed + 1,
        verbose=verbose,
    )
    after_projected = after_state["stitched_projected"]
    after_gmm = after_state["compatibility_gmm"]
    after_metrics = after_state["averaged_metrics"]
    after_metrics["cv_dim"] = int(after_state["compatibility_cv_dim"])
    hidalgo_stats["segment_cv_dims"] = after_state["segment_cv_dims"]
    hidalgo_stats["after_segment_sizes"] = after_state["segment_sizes"]
    hidalgo_stats["after_segment_indices"] = after_state["segment_indices"]
    hidalgo_stats["after_compatibility_cv_dim"] = after_state["compatibility_cv_dim"]
    hidalgo_stats["after_compatibility_active_clusters"] = int((after_gmm["cluster_counts"] > 0).sum().item())
    hidalgo_stats["after_metrics_mode"] = after_state.get("after_metrics_mode", "single clustering")
    hidalgo_stats["before_metrics"] = before_metrics
    hidalgo_stats["after_metrics"] = after_metrics
    if verbose:
        print(
            "after Hidalgo segment cv_dims="
            f"{after_state['segment_cv_dims']} sizes={after_state['segment_sizes']}"
        )
    if verbose:
        print(f"before: active={before_metrics['active_clusters']}/{requested_n_components}")
        print(f"after:  active={after_metrics['active_clusters']}/{requested_n_components}")

    plot_path_obj = None if plot_path is None else Path(plot_path)
    if plot_path_obj is not None:
        plot_path_obj.parent.mkdir(parents=True, exist_ok=True)
        _save_hidalgo_plot(
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            before_gmm=before_gmm,
            after_gmm=after_gmm,
            hidalgo_stats=hidalgo_stats,
            plot_path=plot_path_obj,
            layer_name=layer_name,
            requested_n_components=requested_n_components,
            cluster_population_top_k=cluster_population_top_k,
        )

    history = {
        "stage": ["before", "after_hidalgo"],
        "cv_dim": [initial_cv_dim, discovered_cv_dim],
        "requested_n_components": [requested_n_components, requested_n_components],
        "active_clusters": [
            before_metrics["active_clusters"],
            after_metrics["active_clusters"],
        ],
        "hidalgo": hidalgo_stats,
    }
    return {
        "optimized_reduct_m": full_reduct_m.clone(),
        "optimized_projection": full_reduct_m.clone(),
        "optimized_cv_dim": discovered_cv_dim,
        "best_cv_dim": discovered_cv_dim,
        "initial_cv_dim": initial_cv_dim,
        "cv_dim_candidates": [initial_cv_dim, discovered_cv_dim],
        "before_projected": before_projected.clone(),
        "after_projected": after_projected.clone(),
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "before_gmm": _snapshot_gmm_state(before_gmm),
        "after_gmm": _snapshot_gmm_state(after_gmm) if after_gmm is not None else None,
        "history": history,
        "plot_path": plot_path_obj,
        "n_components": requested_n_components,
        "hidalgo": hidalgo_stats,
        "hidalgo_dimension": float(hidalgo_stats["dimension"]),
        "hidalgo_cv_dim": discovered_cv_dim,
    }

@torch.no_grad()
def hidalgo_dimension(
    data,
    fraction=0.95,
    k=2, q=3, zeta=0.8,
    n_iter=200, n_restarts=6, tol=1e-5,
    confidence_threshold=0.8,
    min_segment_weight=1e-3,
    burn_in=0.9,
    sampling_rate=10,
    alpha_shape=1.0,
    alpha_rate=1.0,
    neighbor_cache=None,
    return_diagnostics=False,
    verbose=False,
    seed=29, 
):
    """
    Estimate local intrinsic dimensions with a Gibbs sampler over:
    - latent segment assignments zi
    - mixture weights p
    - segment dimensions dk

    The local-ID likelihood is based on the ratio mu = r2 / r1:
        p(mu | d) = d * mu^(-d-1), 
    and neighbor consistency is encouraged through a Potts-like term weighted by zeta.
    """
    neighbor_cache = _get_hidalgo_neighbor_cache(
        data=data,
        q=q,
        neighbor_cache=neighbor_cache,
    )
    nearest_distances = neighbor_cache["nearest_distances"]
    nearest_indices = neighbor_cache["nearest_indices"]

    r1 = nearest_distances[:, 1]
    r2 = nearest_distances[:, 2]
    eps = torch.finfo(nearest_distances.dtype).eps
    zero_mask = r1 <= eps
    degenerate_mask = torch.isclose(r1, r2, atol=eps, rtol=0.0)
    good_mask = ~(zero_mask | degenerate_mask)
    good_indices = torch.where(good_mask)[0]
    n_good = int(good_indices.numel())

    if verbose:
        print(f"Found {int(zero_mask.sum().item())} points for which r1=0 and {int(degenerate_mask.sum().item())} points for which r1=r2")
        print(f"Fraction good points: {n_good / max(1, data.shape[0]):.6f}.")

    mu = (r2[good_mask] / r1[good_mask]).clamp_min(1.0 + 1e-12)
    log_mu = torch.log(mu.to(dtype=torch.float64))
    local_dim = log_mu.reciprocal().clamp(min=1e-6, max=1e6)
    npoints = n_good
    fit_mask = torch.ones(n_good, dtype=torch.bool, device=data.device)
    if int(fit_mask.sum().item()) < k:
        raise ValueError("Not enough retained Hidalgo points.")

    local_neighbor_indices = nearest_indices[good_mask, 1 : min(nearest_indices.shape[1], q + 1)]
    good_lookup = torch.full((data.shape[0],), -1, dtype=torch.long, device=data.device)
    good_lookup[good_indices] = torch.arange(n_good, device=data.device)
    good_neighbor_indices = good_lookup[local_neighbor_indices]
    fit_indices = torch.where(fit_mask)[0]
    fit_lookup = torch.full((n_good,), -1, dtype=torch.long, device=data.device)
    fit_lookup[fit_indices] = torch.arange(int(fit_indices.numel()), device=data.device)
    fit_neighbor_indices = fit_lookup[good_neighbor_indices[fit_mask]]
    fit_neighbor_mask = fit_neighbor_indices >= 0
    all_neighbor_fit_indices = fit_lookup[good_neighbor_indices]
    all_neighbor_fit_mask = all_neighbor_fit_indices >= 0
    alpha_dirichlet_tensor = torch.ones((k,), device=log_mu.device, dtype=log_mu.dtype)
    zeta = min(float(zeta), 1.0 - 1e-6)
    potts_strength = log(zeta / (1.0 - zeta))

    segment_state = _run_hidalgo_gibbs_sampler(
        log_mu=log_mu[fit_mask],
        neighbor_indices=fit_neighbor_indices,
        neighbor_mask=fit_neighbor_mask,
        k=k,
        zeta=float(zeta),
        n_iter=n_iter,
        n_restarts=n_restarts,
        tol=tol,
        burn_in=burn_in,
        sampling_rate=sampling_rate,
        alpha_dirichlet=alpha_dirichlet_tensor,
        alpha_shape=float(alpha_shape),
        alpha_rate=float(alpha_rate),
        min_segment_weight=min_segment_weight,
        verbose=verbose,
        seed=seed,
    )

    fitted_dimensions = segment_state["dimensions"]
    fitted_weights = segment_state["weights"]
    effective_dimension = float(torch.dot(fitted_weights, fitted_dimensions).cpu())
    dominant_index = int(torch.argmax(fitted_weights).item())
    dominant_dimension = float(fitted_dimensions[dominant_index].cpu())
    sort_order = torch.argsort(fitted_dimensions)
    sorted_dimensions = fitted_dimensions[sort_order]
    sorted_weights = fitted_weights[sort_order]

    # Fast K-sweep mode:
    # keep only the sampler outputs needed to compare runs by average log posterior.
    
    posterior_fit = segment_state["posterior"]
    posterior_good = _infer_hidalgo_posteriors_for_all_points(
        log_mu=log_mu,
        fit_mask=fit_mask,
        posterior_fit=posterior_fit,
        weights=fitted_weights,
        dimensions=fitted_dimensions,
        neighbor_indices_to_fit=all_neighbor_fit_indices,
        neighbor_mask_to_fit=all_neighbor_fit_mask,
        potts_strength=potts_strength,
    )
    
    posterior_all = torch.zeros((data.shape[0], k), dtype=data.dtype, device=data.device)
    posterior_all[good_mask] = posterior_good.to(dtype=data.dtype)
    fallback = fitted_weights.unsqueeze(0).expand(int((~good_mask).sum().item()), -1)
    if fallback.numel() > 0:
        posterior_all[~good_mask] = fallback.to(dtype=data.dtype)
    
    assignments_all = posterior_all.argmax(dim=1)
    confidence_all = posterior_all.max(dim=1).values
    z_assignments = assignments_all + 1
    z_assignments = z_assignments.to(dtype=torch.long)
    z_assignments[confidence_all < float(confidence_threshold)] = 0
    
    confident_mask = z_assignments > 0
    confident_count = int(confident_mask.sum().item())
    segment_counts = torch.bincount(assignments_all, minlength=k)
    confident_weights = (
        posterior_all[confident_mask].mean(dim=0) if confident_count > 0 else fitted_weights
    )

    stats = {
        "dimension": effective_dimension,
        "dominant_dimension": dominant_dimension,
        "segment_dimensions": fitted_dimensions.cpu(),
        "segment_weights": fitted_weights.cpu(),
        "sorted_segment_dimensions": sorted_dimensions.cpu(),
        "sorted_segment_weights": sorted_weights.cpu(),
        "fraction": float(fraction),
        "n_samples": int(data.shape[0]),
        "n_good_points": int(n_good),
        "n_regression_points": int(npoints),
        "ignored_zero_first_neighbor": int(zero_mask.sum().item()),
        "ignored_equal_neighbors": int(degenerate_mask.sum().item()),
        "confident_count": int(confident_count),
        "confident_fraction": float(confident_count / max(1, data.shape[0])),
        "confident_weights": confident_weights.cpu(),
        "segment_counts": segment_counts.cpu(),
        "assignments": z_assignments.cpu(),
        "raw_assignments": assignments_all.cpu(),
        "assignment_confidence": confidence_all.cpu(),
        "k": int(k),
        "q": int(q),
        "zeta": float(zeta),
        "potts_strength": float(potts_strength),
        "posterior_samples": int(segment_state["n_saved_samples"]),
        "burn_in": float(burn_in),
        "sampling_rate": int(sampling_rate),
        "best_objective": float(segment_state["mean_log_posterior"]),
        "n_iter_run": int(segment_state["n_iter_run"]),
        "x": log_mu.cpu(),
        "y": local_dim.cpu(),
        "neighbor_cache": neighbor_cache,
    }
    if verbose:
        print(
            f"Hidalgo segment dimensions: "
            f"{[round(float(v), 4) for v in fitted_dimensions.cpu().tolist()]}"
        )
        print(
            f"Hidalgo segment weights: "
            f"{[round(float(v), 4) for v in fitted_weights.cpu().tolist()]}"
        )
        print(
            f"Hidalgo effective dimension: {stats['dimension']:.6f} | "
            f"dominant dimension: {stats['dominant_dimension']:.6f}"
        )
        print(
            f"Confident assignments: {stats['confident_count']}/{stats['n_samples']} "
            f"({stats['confident_fraction']:.2%})"
        )
    if return_diagnostics:
        return stats
    return stats["dimension"]


def _get_hidalgo_neighbor_cache(data, q, neighbor_cache=None):
    required_k = min(data.shape[0], int(q) + 1)

    if neighbor_cache is not None:
        cached_q = int(neighbor_cache["q"])
        cached_n_samples = int(neighbor_cache["n_samples"])
        if cached_q != int(q):
            raise ValueError(
                f"neighbor_cache was built with q={cached_q}, but received q={int(q)}."
            )
        if cached_n_samples != int(data.shape[0]):
            raise ValueError(
                "neighbor_cache sample count does not match data. "
                f"Got cache n_samples={cached_n_samples} and data.shape[0]={int(data.shape[0])}."
            )

        nearest_distances = neighbor_cache["nearest_distances"].to(device=data.device)
        nearest_indices = neighbor_cache["nearest_indices"].to(device=data.device)
        if nearest_distances.shape[1] < required_k:
            raise ValueError(
                "neighbor_cache does not contain enough nearest neighbors for this q. "
                f"Need {required_k}, got {nearest_distances.shape[1]}."
            )
        return {
            "q": int(q),
            "n_samples": int(data.shape[0]),
            "nearest_distances": nearest_distances[:, :required_k],
            "nearest_indices": nearest_indices[:, :required_k],
        }

    distances = torch.cdist(data, data, p=2)
    nearest_distances, nearest_indices = torch.topk(
        distances,
        k=required_k,
        dim=1,
        largest=False,
    )
    return {
        "q": int(q),
        "n_samples": int(data.shape[0]),
        "nearest_distances": nearest_distances.clone(),
        "nearest_indices": nearest_indices.clone(),
    }


def _run_hidalgo_gibbs_sampler(
    log_mu,
    neighbor_indices,
    neighbor_mask,
    k, zeta, n_iter,
    n_restarts, tol,
    burn_in,
    sampling_rate,
    alpha_dirichlet,
    alpha_shape, alpha_rate,
    min_segment_weight,
    verbose,
    seed,
):
    best_state = None
    # points next to each other are encouraged to have the same ID
    zeta = min(float(zeta), 1.0 - 1e-6)
    potts_strength = log(zeta / (1.0 - zeta))
    n_points = int(log_mu.shape[0])
    burn_in_steps = min(max(0, int(n_iter * float(burn_in))), max(0, n_iter - 1))
    q = int(neighbor_indices.shape[1]) if neighbor_indices.ndim == 2 else 0
    log_zpart_cache = _build_hidalgo_log_zpart_cache(
        total_points=int(log_mu.shape[0]),
        zeta=float(zeta),
        q=q,
        device=log_mu.device,
        dtype=log_mu.dtype,
    )

    torch.manual_seed(int(seed))

    for restart_idx in range(max(1, n_restarts)):
        restart_seed = int(seed) + restart_idx
        torch.manual_seed(restart_seed)

        state = _initialize_hidalgo_chain(
            log_mu=log_mu,
            k=k,
        )

        # K-sweep mode:
        # keep only what is needed to compare runs via mean log posterior.
        posterior_counts = torch.zeros((n_points, k), device=log_mu.device, dtype=log_mu.dtype)
        weight_sum = torch.zeros((k,), device=log_mu.device, dtype=log_mu.dtype)
        dimension_sum = torch.zeros((k,), device=log_mu.device, dtype=log_mu.dtype)
        saved_log_posteriors = []
        saved_samples = 0
        last_saved_sample = None
        prev_dimension_mean = None
        stable_saves = 0

        for iteration_idx in range(max(1, n_iter)):
            assignment_counts = torch.bincount(state["assignments"], minlength=k).to(dtype=log_mu.dtype)
            state["weights"] = _sample_hidalgo_weights(
                assignments=state["assignments"],
                k=k,
                alpha_dirichlet=alpha_dirichlet,
                min_segment_weight=min_segment_weight,
                counts=assignment_counts,
            )
            state["dimensions"] = _sample_hidalgo_dimensions(
                log_mu=log_mu,
                assignments=state["assignments"],
                k=k,
                alpha_shape=alpha_shape,
                alpha_rate=alpha_rate,
                counts=assignment_counts,
            )
            _sample_hidalgo_assignments_in_batches(
                state=state,
                log_mu=log_mu,
                neighbor_indices=neighbor_indices,
                neighbor_mask=neighbor_mask,
                potts_strength=potts_strength,
                batch_size=32,
            )

            should_save = iteration_idx >= burn_in_steps
            if should_save and ((iteration_idx - burn_in_steps) % int(sampling_rate) == 0):
                sorted_sample = _sorted_hidalgo_sample(
                    assignments=state["assignments"],
                    weights=state["weights"],
                    dimensions=state["dimensions"],
                    k=k,
                )
                last_saved_sample = sorted_sample
                posterior_counts += torch.nn.functional.one_hot(
                    sorted_sample["assignments"],
                    num_classes=k,
                ).to(dtype=posterior_counts.dtype)
                weight_sum += sorted_sample["weights"]
                dimension_sum += sorted_sample["dimensions"]
                saved_log_posteriors.append(
                    _hidalgo_reference_log_posterior_from_assignments(
                        log_mu=log_mu,
                        assignments=sorted_sample["assignments"],
                        weights=sorted_sample["weights"],
                        dimensions=sorted_sample["dimensions"],
                        neighbor_indices=neighbor_indices,
                        neighbor_mask=neighbor_mask,
                        zeta=float(zeta),
                        potts_strength=float(potts_strength),
                        log_zpart_cache=log_zpart_cache,
                    )
                )
                saved_samples += 1

                current_dimension_mean = dimension_sum / saved_samples
                if prev_dimension_mean is not None:
                    max_shift = torch.max(torch.abs(current_dimension_mean - prev_dimension_mean))
                    if float(max_shift.cpu()) <= tol:
                        stable_saves += 1
                    else:
                        stable_saves = 0
                prev_dimension_mean = current_dimension_mean.clone()
                
                if stable_saves >= 3:
                    break

        if saved_samples <= 0:
            sorted_sample = _sorted_hidalgo_sample(
                assignments=state["assignments"],
                weights=state["weights"],
                dimensions=state["dimensions"],
                k=k,
            )
            last_saved_sample = sorted_sample
            posterior_counts += torch.nn.functional.one_hot(
                sorted_sample["assignments"],
                num_classes=k,
            ).to(dtype=posterior_counts.dtype)
            weight_sum += sorted_sample["weights"]
            dimension_sum += sorted_sample["dimensions"]
            saved_log_posteriors.append(
                _hidalgo_reference_log_posterior_from_assignments(
                    log_mu=log_mu,
                    assignments=sorted_sample["assignments"],
                    weights=sorted_sample["weights"],
                    dimensions=sorted_sample["dimensions"],
                    neighbor_indices=neighbor_indices,
                    neighbor_mask=neighbor_mask,
                    zeta=float(zeta),
                    potts_strength=float(potts_strength),
                    log_zpart_cache=log_zpart_cache,
                )
            )
            saved_samples = 1

        posterior = posterior_counts / saved_samples
        weights = weight_sum / saved_samples
        weights = weights / weights.sum().clamp_min(1e-12)
        dimensions = dimension_sum / saved_samples
        mean_log_posterior = sum(saved_log_posteriors) / len(saved_log_posteriors)

        candidate_state = {
            "posterior": posterior,
            "weights": weights,
            "dimensions": dimensions,
            "mean_log_posterior": float(mean_log_posterior),
            "n_saved_samples": int(saved_samples),
            "n_iter_run": int(iteration_idx + 1),
        }
        if best_state is None or candidate_state["mean_log_posterior"] > best_state["mean_log_posterior"]:
            best_state = candidate_state
        if verbose:
            print(
                f"Hidalgo restart {restart_idx + 1}/{max(1, n_restarts)}: "
                f"mean_log_posterior={candidate_state['mean_log_posterior']:.6f}, "
                f"iterations={candidate_state['n_iter_run']}, "
                f"saved_samples={candidate_state['n_saved_samples']}"
            )

    return best_state


def _initialize_hidalgo_chain(log_mu, k):
    local_dim = log_mu.reciprocal().clamp(min=1e-6, max=1e6)
    quantiles = torch.linspace(0.0, 1.0, steps=k + 2, device=log_mu.device, dtype=log_mu.dtype)[1:-1]
    dimensions = torch.quantile(local_dim, quantiles)
    if dimensions.ndim == 0:
        dimensions = dimensions.unsqueeze(0)
    if dimensions.numel() < k:
        dimensions = dimensions.repeat(k)[:k]
    noise = 0.05 * torch.randn((k,), device=log_mu.device, dtype=log_mu.dtype)
    dimensions = (dimensions + noise).clamp(min=1e-3)
    weights = torch.full((k,), 1.0 / k, device=log_mu.device, dtype=log_mu.dtype)
    assignments = torch.bucketize(local_dim, torch.quantile(local_dim, quantiles), right=False)
    assignments = assignments.clamp(max=k - 1).to(dtype=torch.long)
    return {
        "assignments": assignments,
        "weights": weights,
        "dimensions": dimensions,
    }

def _sample_hidalgo_weights(assignments, k, alpha_dirichlet, min_segment_weight, counts=None):
    if counts is None:
        counts = torch.bincount(assignments, minlength=k).to(dtype=alpha_dirichlet.dtype)
    else:
        counts = counts.to(dtype=alpha_dirichlet.dtype)
    alpha_post = (alpha_dirichlet + counts).clamp_min(1e-6)
    gamma_samples = torch._standard_gamma(alpha_post)
    weights = gamma_samples / gamma_samples.sum().clamp_min(1e-12)
    weights = weights.clamp_min(float(min_segment_weight))
    return weights / weights.sum().clamp_min(1e-12)


def _sample_hidalgo_dimensions(log_mu, assignments, k, alpha_shape, alpha_rate, counts=None):
    if counts is None:
        counts = torch.bincount(assignments, minlength=k).to(dtype=log_mu.dtype)
    else:
        counts = counts.to(dtype=log_mu.dtype)
    log_mu_sums = torch.zeros((k,), device=log_mu.device, dtype=log_mu.dtype)
    log_mu_sums.scatter_add_(0, assignments, log_mu)
    post_shape = counts + float(alpha_shape)
    post_rate = log_mu_sums + float(alpha_rate)
    sampled = torch._standard_gamma(post_shape.clamp_min(1e-6))
    dimensions = sampled / post_rate.clamp_min(1e-12)
    return dimensions.clamp(min=1e-3, max=1e6)


def _sample_hidalgo_assignments_in_batches(
    state,
    log_mu,
    neighbor_indices,
    neighbor_mask,
    potts_strength,
    batch_size=1,
):

    assignments = state["assignments"]
    n_points = int(assignments.shape[0])
    cached_terms = _compute_hidalgo_cached_log_terms(
        log_mu=log_mu,
        weights=state["weights"],
        dimensions=state["dimensions"],
    )

    for batch_start in range(0, n_points, int(batch_size)):
        batch_end = min(batch_start + int(batch_size), n_points)
        batch_slice = slice(batch_start, batch_end)
        log_prob = cached_terms[batch_slice].clone()

        if potts_strength > 0.0 and neighbor_indices.numel() > 0:
            current_neighbors = neighbor_indices[batch_slice]
            current_mask = neighbor_mask[batch_slice]
            safe_neighbors = current_neighbors.clamp_min(0)
            neighbor_assignments = assignments[safe_neighbors]
            neighbor_counts = torch.zeros(
                (neighbor_assignments.shape[0], state["weights"].shape[0]),
                device=log_prob.device,
                dtype=log_prob.dtype,
            )
            neighbor_counts.scatter_add_(
                1,
                neighbor_assignments,
                current_mask.to(dtype=log_prob.dtype),
            )
            log_prob = log_prob + float(potts_strength) * neighbor_counts

        probs = torch.softmax(log_prob, dim=1)
        assignments[batch_slice] = torch.multinomial(probs, num_samples=1).squeeze(1)


def _compute_hidalgo_cached_log_terms(log_mu, weights, dimensions):
    log_weights = torch.log(weights.clamp_min(1e-12)).unsqueeze(0)
    log_dimensions = torch.log(dimensions.clamp_min(1e-12)).unsqueeze(0)
    dimension_log_mu = log_mu.unsqueeze(1) * dimensions.unsqueeze(0)
    return log_weights + log_dimensions - dimension_log_mu


def _sorted_hidalgo_sample(assignments, weights, dimensions, k):
    sort_order = torch.argsort(dimensions)
    inverse_order = torch.empty_like(sort_order)
    inverse_order[sort_order] = torch.arange(k, device=sort_order.device)
    sorted_assignments = inverse_order[assignments]
    return {
        "assignments": sorted_assignments,
        "weights": weights[sort_order],
        "dimensions": dimensions[sort_order],
    }


def _hidalgo_reference_log_posterior_from_assignments(
    log_mu,
    assignments,
    weights,
    dimensions,
    neighbor_indices,
    neighbor_mask,
    zeta,
    potts_strength=None,
    log_zpart_cache=None,
):
    """
    score = how well ratios fit the assined dimension
          + encoragement for neighbours to have the same assignment
          - normalization term 
          
    (`lik1` in micheleallegra/Hidalgo/python/gibbs.c).
    """
    assigned_dimensions = dimensions[assignments]
    assigned_weights = weights[assignments].clamp_min(1e-12)
    base_log_likelihood = (
        torch.log(assigned_weights)
        + torch.log(assigned_dimensions.clamp_min(1e-12))
        - (assigned_dimensions + 1.0) * log_mu
    ).sum()

    same_neighbor_count = torch.tensor(0.0, device=log_mu.device, dtype=log_mu.dtype)
    if neighbor_indices.numel() > 0:
        safe_indices = neighbor_indices.clamp_min(0)
        neighbor_assignments = assignments[safe_indices]
        # How many neighbors agree in their segment assignment?
        agreement = (neighbor_assignments == assignments.unsqueeze(1)).to(dtype=log_mu.dtype)
        agreement = agreement * neighbor_mask.to(dtype=log_mu.dtype)
        same_neighbor_count = agreement.sum()

    segment_counts = torch.bincount(assignments, minlength=weights.shape[0])
    if log_zpart_cache is None:
        q = int(neighbor_indices.shape[1]) if neighbor_indices.ndim == 2 else 0
        log_zpart_cache = _build_hidalgo_log_zpart_cache(
            total_points=int(assignments.shape[0]),
            zeta=float(zeta),
            q=q,
            device=weights.device,
            dtype=weights.dtype,
        )
    normalization_term = (
        segment_counts.to(dtype=log_zpart_cache.dtype) * log_zpart_cache[segment_counts]
    ).sum()

    if potts_strength is None:
        potts_strength = log(float(zeta) / (1.0 - float(zeta)))
    score = base_log_likelihood + float(potts_strength) * same_neighbor_count - normalization_term
    return float(score.cpu())


def _build_hidalgo_log_zpart_cache(total_points, zeta, q, device, dtype):
    cache = torch.zeros((int(total_points) + 1,), device=device, dtype=dtype)
    for segment_size in range(1, int(total_points) + 1):
        cache[segment_size] = _hidalgo_log_zpart(
            total_points=total_points,
            segment_size=segment_size,
            zeta=zeta,
            q=q,
        )
    return cache


def _hidalgo_log_zpart(total_points, segment_size, zeta, q):
    if q <= 0:
        return 0.0

    log_terms = []
    same_pool = max(0, int(segment_size) - 1)
    diff_pool = max(0, int(total_points) - int(segment_size))
    for same_neighbors in range(q + 1):
        diff_neighbors = q - same_neighbors
        if same_neighbors > same_pool or diff_neighbors > diff_pool:
            continue
        log_term = (
            _log_binomial_coefficient(same_pool, same_neighbors)
            + _log_binomial_coefficient(diff_pool, diff_neighbors)
            + same_neighbors * log(float(zeta))
            + diff_neighbors * log(1.0 - float(zeta))
        )
        log_terms.append(log_term)

    if not log_terms:
        return 0.0

    max_log_term = max(log_terms)
    return max_log_term + log(sum(exp(term - max_log_term) for term in log_terms))


def _log_binomial_coefficient(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(float(n + 1)) - lgamma(float(k + 1)) - lgamma(float(n - k + 1))


def _infer_hidalgo_posteriors_for_all_points(
    log_mu,
    fit_mask,
    posterior_fit,
    weights,
    dimensions,
    neighbor_indices_to_fit,
    neighbor_mask_to_fit,
    potts_strength,
):
    k = int(weights.shape[0])
    posterior_good = torch.zeros((log_mu.shape[0], k), device=log_mu.device, dtype=log_mu.dtype)
    posterior_good[fit_mask] = posterior_fit

    base_log_prob = torch.log(weights.clamp_min(1e-12)).unsqueeze(0)
    base_log_prob = base_log_prob + torch.log(dimensions.clamp_min(1e-12)).unsqueeze(0)
    base_log_prob = base_log_prob - log_mu.unsqueeze(1) * dimensions.unsqueeze(0)

    unresolved_mask = ~fit_mask
    if bool(unresolved_mask.any().cpu().item()):
        unresolved_indices = torch.where(unresolved_mask)[0]
        unresolved_log_prob = base_log_prob[unresolved_indices]
        unresolved_neighbor_indices = neighbor_indices_to_fit[unresolved_indices]
        unresolved_neighbor_mask = neighbor_mask_to_fit[unresolved_indices]

        if potts_strength > 0.0 and unresolved_neighbor_indices.numel() > 0:
            safe_indices = unresolved_neighbor_indices.clamp_min(0)
            neighbor_post = posterior_fit[safe_indices]
            neighbor_post = neighbor_post * unresolved_neighbor_mask.unsqueeze(-1).to(dtype=neighbor_post.dtype)
            counts = unresolved_neighbor_mask.sum(dim=1, keepdim=True).clamp_min(1)
            smoothed = neighbor_post.sum(dim=1) / counts
            unresolved_log_prob = unresolved_log_prob + float(potts_strength) * smoothed

        posterior_good[unresolved_indices] = torch.softmax(unresolved_log_prob, dim=1)

    return posterior_good

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
    if float(weights.sum().cpu()) <= 0.0:
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
        "weights": weights.clone(),
        "means": means.clone(),
        "variances": variances.clone(),
        "assignments": assignments.clone(),
        "cluster_counts": cluster_counts.clone(),
        "cluster_marginal_profile": posterior.mean(dim=0).clone(),
        "max_assignment_probabilities": posterior.max(dim=1).values.clone(),
        "nll": (-log_norm.sum()).clone(),
    }


def _compute_hidalgo_metrics(
    projected,
    gmm_state,
    cv_dim,
    requested_n_components,
    labels=None,
    coverage_threshold=0.8,
    n_classes=None,
):
    counts = gmm_state["cluster_counts"]
    active_clusters = int((counts > 0).sum().item())

    n_samples = max(1, projected.shape[0])
    nll = _finite_or_nan(float(gmm_state["nll"].cpu()))
    complexity = _gmm_num_parameters(
        n_components=requested_n_components,
        n_features=int(projected.shape[1]),
    )
    bic_penalty = float(complexity * log(max(2, n_samples)))
    bic = _finite_or_nan(2.0 * nll + bic_penalty)
    silhouette = _finite_or_nan(
        float(_silhouette_score(projected, gmm_state["assignments"].long()).cpu())
    )

    metrics = {
        "cv_dim": int(cv_dim),
        "requested_n_components": int(requested_n_components),
        "active_clusters": int(active_clusters),
        "silhouette": silhouette,
        "md_col_mean": _mahalanobis_col_mean(gmm_state),
        "cluster_size_imbalance_ratio": _cluster_size_imbalance_ratio(counts),
        "bic": bic,
        "nll": nll,
    }

    if labels is not None:
        ami = _finite_or_nan(
            adjusted_mutual_info_score(
                labels.cpu().numpy(),
                gmm_state["assignments"].long().cpu().numpy(),
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


def _compute_hidalgo_max_dim_after_state(
    h_data,
    reduct_m,
    requested_n_components,
    hidalgo_stats,
    labels,
    coverage_threshold,
    n_classes,
    full_rank,
    seed,
    verbose,
):
    assignments = torch.as_tensor(hidalgo_stats["assignments"], device=h_data.device, dtype=torch.long)
    segment_dimensions = torch.as_tensor(
        hidalgo_stats["segment_dimensions"],
        device=h_data.device,
        dtype=h_data.dtype,
    )
    if assignments.shape[0] != h_data.shape[0]:
        raise ValueError("Hidalgo assignments must match h_data sample count.")

    segment_cv_dims = [
        int(max(1, min(full_rank, ceil(float(dim.cpu())))))
        for dim in segment_dimensions
    ]
    segment_sizes = []
    segment_indices = []

    active_segment_cv_dims = []
    for segment_idx, segment_cv_dim in enumerate(segment_cv_dims):
        # Hidalgo assignments are 1-based for confident segments; 0 is reserved
        # for unconfident points.
        mask = assignments == int(segment_idx + 1)
        segment_size = int(mask.sum().item())
        if segment_size <= 0:
            continue
        segment_sizes.append(segment_size)
        segment_indices.append(int(segment_idx))
        active_segment_cv_dims.append(int(segment_cv_dim))

    if active_segment_cv_dims:
        compatibility_cv_dim = max(active_segment_cv_dims)
    else:
        compatibility_cv_dim = int(max(1, min(full_rank, ceil(float(hidalgo_stats["dimension"])))))

    stitched_projected = _project_with_cv_dim(
        h_data=h_data,
        reduct_m=reduct_m,
        cv_dim=compatibility_cv_dim,
    )
    compatibility_gmm = _fit_stable_gmm(
        projected=stitched_projected,
        n_components=requested_n_components,
        seed=None if seed is None else seed + 197,
        verbose=verbose,
    )
    averaged_metrics = _compute_hidalgo_metrics(
        projected=stitched_projected,
        gmm_state=compatibility_gmm,
        cv_dim=compatibility_cv_dim,
        requested_n_components=requested_n_components,
        labels=labels,
        coverage_threshold=coverage_threshold,
        n_classes=n_classes,
    )
    return {
        "segment_cv_dims": segment_cv_dims,
        "segment_sizes": segment_sizes,
        "segment_indices": segment_indices,
        "stitched_projected": stitched_projected,
        "compatibility_gmm": compatibility_gmm,
        "compatibility_cv_dim": compatibility_cv_dim,
        "averaged_metrics": averaged_metrics,
        "after_metrics_mode": "single clustering (max segment dim)",
    }


def _save_hidalgo_plot(
    before_metrics, after_metrics,
    before_gmm, after_gmm,
    hidalgo_stats,
    plot_path,
    layer_name,
    requested_n_components,
    cluster_population_top_k,
):
    fig = plt.figure(figsize=(18, 16))
    grid = fig.add_gridspec(
        3,2,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.72, 1.0, 1.35],
        hspace=0.45, wspace=0.25,
    )

    ax_summary = fig.add_subplot(grid[0, :])
    ax_profile = fig.add_subplot(grid[1, :])
    ax_before_pop = fig.add_subplot(grid[2, 0])
    ax_after_pop = fig.add_subplot(grid[2, 1])

    fig.suptitle(f"Layer: {layer_name}", fontsize=16)

    _render_hidalgo_summary_panel(
        ax=ax_summary,
        hidalgo_stats=hidalgo_stats,
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
        title="Cluster populations: Hidalgo projection",
        gmm_state=after_gmm,
        cv_dim=after_metrics["cv_dim"],
        requested_n_components=requested_n_components,
        top_k=cluster_population_top_k,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_hidalgo_summary_panel(ax, hidalgo_stats, before_metrics, after_metrics, requested_n_components):
    ax.axis("off")
    segment_cv_dims = [int(v) for v in hidalgo_stats.get("segment_cv_dims", [])]
    segment_sizes = [int(v) for v in hidalgo_stats.get("after_segment_sizes", [])]
    segment_indices = [int(v) for v in hidalgo_stats.get("after_segment_indices", [])]
    segment_lines = []
    if segment_indices and segment_sizes:
        for seg_idx, seg_size in zip(segment_indices, segment_sizes):
            seg_dim = int(segment_cv_dims[seg_idx]) if seg_idx < len(segment_cv_dims) else None
            segment_lines.append(f"seg {seg_idx + 1}: dim={seg_dim}, size={seg_size}")
    else:
        for idx, seg_dim in enumerate(segment_cv_dims):
            segment_lines.append(f"seg {idx + 1}: dim={seg_dim}")

    summary_lines = [
        f"Original cv_dim: {hidalgo_stats['initial_cv_dim']}",
        f"Used Hidalgo cv_dim: {hidalgo_stats['cv_dim']}",
        f"Fixed GMM n_components: {requested_n_components}",
        f"Hidalgo params: K={hidalgo_stats['k']}, q={hidalgo_stats['q']}, zeta={hidalgo_stats['zeta']:.3f}",
        f"Potts strength: {hidalgo_stats['potts_strength']:.3f}",
        f"After metrics mode: {hidalgo_stats.get('after_metrics_mode', 'single clustering')}",
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
        (
            f"Good points: {hidalgo_stats['n_good_points']}/{hidalgo_stats['n_samples']} | "
            f"Fit points: {hidalgo_stats['n_regression_points']}"
        ),
        (
            f"Confident assignments: {hidalgo_stats['confident_count']}/{hidalgo_stats['n_samples']} "
            f"({hidalgo_stats['confident_fraction']:.2%})"
        ),
        "Segments:",
        *segment_lines,
    ]
    ax.text(
        0.01, 0.97,
        "\n".join(line for line in summary_lines if line is not None),
        va="top", ha="left",
        family="monospace",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7", edgecolor="#d0d0d0"),
    )
    ax.set_title("Hidalgo summary", loc="left", fontsize=13)


def _render_cluster_marginal_profile(ax, before_gmm, after_gmm):
    before_profile = _sorted_cluster_profile(before_gmm)
    after_profile = _sorted_cluster_profile(after_gmm) if after_gmm is not None else []

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

    ax.set_title("Cluster posterior profile before/after Hidalgo selection")
    ax.set_xlabel("Components ordered by descending posterior mass")
    ax.set_ylabel("Average posterior mass per component")
    ax.grid(True, axis="both", alpha=0.25)
    ax.legend()


def _render_cluster_population_panel(ax, title, gmm_state, cv_dim, requested_n_components, top_k):
    ax.axis("off")
    ax.set_title(title)
    ax.text(
        0.01, 0.99,
        _cluster_population_summary(
            gmm_state=gmm_state,
            cv_dim=cv_dim,
            requested_n_components=requested_n_components,
            top_k=top_k,
        ),
        va="top", ha="left",
        family="monospace",
        fontsize=10,
    )


def _cluster_population_summary(gmm_state, cv_dim, requested_n_components, top_k):
    counts = gmm_state["cluster_counts"].cpu().tolist()
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
        torch.as_tensor(gmm_state["cluster_marginal_profile"]).float().cpu(),
        descending=True,
    ).values
    return profile.tolist()


def _snapshot_gmm_state(gmm_state):
    return {
        "weights": gmm_state["weights"].clone(),
        "means": gmm_state["means"].clone(),
        "variances": gmm_state["variances"].clone(),
        "assignments": gmm_state["assignments"].clone(),
        "cluster_counts": gmm_state["cluster_counts"].clone(),
        "cluster_marginal_profile": gmm_state["cluster_marginal_profile"].clone(),
        "max_assignment_probabilities": gmm_state["max_assignment_probabilities"].clone(),
        "nll": gmm_state["nll"].clone(),
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
    return float(col_mean.mean().cpu())


def _asymmetric_mahalanobis_matrix(gmm_state):
    if gmm_state is None:
        return None

    means = gmm_state.get("means")
    variances = gmm_state.get("variances")
    if means is None or variances is None:
        return None

    means = torch.as_tensor(means).float().cpu()
    variances = torch.as_tensor(variances).float().cpu().clamp_min(1e-12)

    inv_var = variances.reciprocal()
    weighted_means = means * inv_var
    t1 = means.pow(2) @ inv_var.T
    t2 = means @ weighted_means.T
    t3 = (means.pow(2) * inv_var).sum(dim=1)
    sq = t1 - 2.0 * t2 + t3.unsqueeze(0)
    sq.clamp_(min=0.0)
    return sq.sqrt()


def _cluster_size_imbalance_ratio(counts):
    counts = torch.as_tensor(counts).float()
    positive_counts = counts[counts > 0]
    if positive_counts.numel() == 0:
        return float("nan")

    min_count = positive_counts.min()
    if float(min_count.cpu()) <= 0.0:
        return float("nan")

    max_count = positive_counts.max()
    return float((max_count / min_count).cpu())


def _format_metric_value(value):
    if value is None:
        return "-"
    value = float(value)
    if value != value:
        return "nan"
    return f"{value:.6f}"


def _gmm_component_log_prob(data, weights, means, variances):
    inv_var = variances.reciprocal()
    log_det = variances.log().sum(dim=1)
    x2_term = data.pow(2) @ inv_var.T
    cross_term = data @ (means * inv_var).T
    mean_term = (means.pow(2) * inv_var).sum(dim=1)
    mahalanobis = x2_term - 2.0 * cross_term + mean_term.unsqueeze(0)
    n_features = data.shape[1]
    return (
        weights.clamp_min(1e-12).log().unsqueeze(0)
        - 0.5 * (n_features * log(2.0 * pi) + log_det.unsqueeze(0) + mahalanobis)
    )
