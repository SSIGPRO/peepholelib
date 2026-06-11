from math import log, pi
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from torchgmm.bayes import GaussianMixture as tGMM
from dpgmm.samplers import DiagCovarianceCollapsedGibbsSampler as DPGMMDiagSampler


def _target_layers_from_kwargs(kwargs):
    target_layers = kwargs.get("target_layers")
    if target_layers is None:
        return None
    if isinstance(target_layers, str):
        return [target_layers]
    return list(target_layers)


def _select_layer_value(value, layer, key, target_layers, shared_ok=False):
    if value is None:
        return None
    if not torch.is_tensor(value) and not isinstance(value, (str, bytes, Path)):
        try:
            return value[layer]
        except (KeyError, TypeError, IndexError):
            pass
    if shared_ok or len(target_layers) == 1:
        return value
    raise ValueError(
        f"{key} must be keyed by layer when optimizing multiple target_layers."
    )


def optimize_projection(**kwargs):
    """
    Optimize a projection matrix initialized from V^T for mixture-model clustering.
    This stage freezes the fitted clustering model and updates only the first
    cv_dim rows of the projection matrix.
    """
    target_layers = _target_layers_from_kwargs(kwargs)
    if target_layers is not None:
        results = {}
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
            results[layer] = optimize_projection(**layer_kwargs)
        return results

    h_data = kwargs["h_data"]
    reduct_m = kwargs["reduct_m"]
    cv_dim = kwargs["cv_dim"]
    loss_name = kwargs.get("loss", "nll").lower()
    n_components = kwargs["n_components"]
    plot_path = kwargs.get("plot_path")
    n_epochs = kwargs.get("n_epochs", 100)
    lr = kwargs.get("lr", 1e-3)
    weight_decay = kwargs.get("weight_decay", 0.0)
    seed = kwargs.get("seed", 29)
    verbose = kwargs.get("verbose", False)
    datasets = kwargs.get("datasets")
    loader = kwargs.get("loader", "train")
    label_key = kwargs.get("label_key", "label")
    layer_name = kwargs.get("layer_name", kwargs.get("layer"))
    coverage_threshold = float(kwargs.get("coverage_threshold", 0.8))
    n_classes = kwargs.get("n_classes", kwargs.get("n_classes"))
    cluster_population_top_k = int(kwargs.get("cluster_population_top_k", 10))
    cluster_method = str(kwargs.get("cluster_method", "gmm")).strip().lower()
    dpgmm_max_clusters = int(kwargs.get("dpgmm_max_clusters", 100)) #attention is not the max
    dpgmm_iterations = int(kwargs.get("dpgmm_iterations", 100))
    gmm_retries = int(kwargs.get("gmm_retries", 10))
    covariance_type = str(kwargs.get("covariance_type", "diag")).strip().lower()

    if loss_name not in {"nll", "bic", "silhouette"}:
        raise ValueError(f"Unknown loss '{loss_name}'. Expected 'nll', 'bic' or 'silhouette'.")
    if cluster_method not in {"gmm", "dpgmm"}:
        raise ValueError("Unknown cluster_method. Expected 'gmm' or 'dpgmm'")

    if h_data.shape[1] != reduct_m.shape[1]:
        raise RuntimeError(
            f"Input dim mismatch: h_data: {h_data.shape[1]}. reduct_m expects {reduct_m.shape[1]}"
        )

    if seed is not None:
        torch.manual_seed(seed)

    device = reduct_m.device
    dtype = reduct_m.dtype
    full_reduct_m = reduct_m.detach().to(device=device, dtype=dtype)
    if cv_dim > full_reduct_m.shape[0]:
        raise RuntimeError(f"cv_dim={cv_dim} exceeds proj rank {full_reduct_m.shape[0]}")

    h_data = h_data.detach().to(device=device, dtype=dtype)
    labels = None if datasets is None else _get_labels_from_dataset(
        datasets=datasets,
        loader=loader,
        label_key=label_key,
        device=device,
    )

    if labels is not None and labels.shape[0] != h_data.shape[0]:
        raise ValueError(
            "act_data and datasets loader must contain the same number of samples. "
            f"Got {h_data.shape[0]} projected samples and {labels.shape[0]} labels."
        )
    active_reduct_m = full_reduct_m[:cv_dim]
    linear = torch.nn.Linear(active_reduct_m.shape[1], active_reduct_m.shape[0], bias=False).to(
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(active_reduct_m.detach())

    optimizer = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=weight_decay)

    with torch.no_grad():
        before_proj = linear(h_data)
        before_gmm = _fit_gmm(
            before_proj,
            n_components=n_components,
            seed=seed,
            cluster_method=cluster_method,
            dpgmm_max_clusters=dpgmm_max_clusters,
            dpgmm_iterations=dpgmm_iterations,
            gmm_retries=gmm_retries,
            covariance_type=covariance_type,
        )

    n_samples = h_data.shape[0]
    effective_n_components = int(before_gmm["weights"].shape[0])
    n_params = _gmm_num_parameters(
        n_components=effective_n_components,
        n_features=before_proj.shape[1],
    )
    before_metrics = _compute_clustering_metrics(
        projected=before_proj,
        gmm_state=before_gmm,
        loss_name=loss_name,
        n_params=n_params,
        n_samples=n_samples,
        seed=seed,
        labels=labels,
        coverage_threshold=coverage_threshold,
        n_classes=n_classes,
    )

    history = {"epoch": [], "objective": [], "nll": [], "bic": []}
    if loss_name == "nll":
        best_eval_objective = before_gmm["nll"]
    elif loss_name == "bic":
        best_eval_objective = _bic_from_nll(before_gmm["nll"], n_params=n_params, n_samples=n_samples)
    else:
        best_eval_objective = _silhouette_score(before_proj, before_gmm["assignments"], seed=seed)

    best_weight = linear.weight.detach().clone()
    frozen_gmm = before_gmm

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        projected = linear(h_data)
        train_log_prob = _gmm_component_log_prob(
            data=projected,
            weights=frozen_gmm["weights"],
            means=frozen_gmm["means"],
            variances=frozen_gmm["variances"],
        )
        train_nll = -torch.logsumexp(train_log_prob, dim=1).sum()
        if loss_name == "nll":
            train_objective = train_nll
        elif loss_name == "bic":
            train_objective = _bic_from_nll(train_nll, n_params=n_params, n_samples=n_samples)
        elif loss_name == "silhouette":
            train_objective = _silhouette_score(projected, frozen_gmm["assignments"], seed=seed)
        else:
            raise ValueError(f"Unknown loss '{loss_name}'.")

        train_loss = -train_objective if loss_name == "silhouette" else train_objective
        if not bool(torch.isfinite(train_loss).detach().cpu().item()):
            train_loss = projected.sum() * 0.0
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            eval_proj = linear(h_data)
            eval_log_prob = _gmm_component_log_prob(
                data=eval_proj,
                weights=frozen_gmm["weights"],
                means=frozen_gmm["means"],
                variances=frozen_gmm["variances"],
            )
            eval_nll = -torch.logsumexp(eval_log_prob, dim=1).sum()
            if loss_name == "nll":
                eval_objective = eval_nll
            elif loss_name == "bic":
                eval_objective = _bic_from_nll(eval_nll, n_params=n_params, n_samples=n_samples)
            elif loss_name == "silhouette":
                eval_objective = _silhouette_score(eval_proj, frozen_gmm["assignments"], seed=seed)
            else:
                raise ValueError(f"Unknown loss '{loss_name}'.")
            eval_bic = _bic_from_nll(eval_nll, n_params=n_params, n_samples=n_samples)

        history["epoch"].append(epoch)
        history["objective"].append(float(eval_objective.detach().cpu()))
        history["nll"].append(float(eval_nll.detach().cpu()))
        history["bic"].append(float(eval_bic.detach().cpu()))

        if verbose:
            print(
                f"epoch={epoch + 1}/{n_epochs}, "
                f"objective={history['objective'][-1]:.6f}, "
                f"nll={history['nll'][-1]:.6f}, "
                f"bic={history['bic'][-1]:.6f}"
            )

        if loss_name == "silhouette":
            improved = eval_objective > best_eval_objective
        else:
            improved = eval_objective < best_eval_objective
        if improved:
            best_eval_objective = eval_objective
            best_weight = linear.weight.detach().clone()

    with torch.no_grad():
        linear.weight.copy_(best_weight)
        optimized_reduct_m = full_reduct_m.clone()
        optimized_reduct_m[:cv_dim] = linear.weight.detach()
        after_proj = linear(h_data)
        if loss_name == "silhouette":
            after_gmm = _evaluate_gmm(after_proj, frozen_gmm, assignments=frozen_gmm["assignments"])
        else:
            after_gmm = _evaluate_gmm(after_proj, frozen_gmm)

    after_metrics = _compute_clustering_metrics(
        projected=after_proj,
        gmm_state=after_gmm,
        loss_name=loss_name,
        n_params=n_params,
        n_samples=n_samples,
        seed=None if seed is None else seed + 49,
        labels=labels,
        coverage_threshold=coverage_threshold,
        n_classes=n_classes,
    )

    if plot_path is not None:
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        projection_cosine_distance = _cosine_distance(full_reduct_m[:cv_dim], optimized_reduct_m[:cv_dim])
        _save_clustering_stats_plot(
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            before_gmm=before_gmm,
            after_gmm=after_gmm,
            plot_path=plot_path,
            loss_name=loss_name,
            layer_name=layer_name,
            summary_text=(
                f"Fixed {cluster_method.upper()}: "
                f"{effective_n_components} active components | Optimizing projection only"
            ),
            cluster_population_top_k=cluster_population_top_k,
            cluster_population_mode="both",
            before_label="Initial projection",
            after_label="Optimized projection",
            improvement_title=f"Projection improvement (frozen {cluster_method.upper()})",
            projection_cosine_distance=projection_cosine_distance,
            marginal_profile_title=(
                f"Cluster marginal posterior profile under frozen {cluster_method.upper()}"
            ),
            before_population_title="Initial projection cluster populations",
            after_population_title="Optimized projection cluster populations",
        )

    return {
        "optimized_reduct_m": optimized_reduct_m,
        "optimized_projection": optimized_reduct_m.detach().clone(),
        "optimized_cv_dim": int(cv_dim),
        "initial_cv_dim": int(cv_dim),
        "before_projected": before_proj.detach().clone(),
        "after_projected": after_proj.detach().clone(),
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "before_gmm": _snapshot_gmm_state(before_gmm, n_params=n_params, n_samples=n_samples),
        "after_gmm": _snapshot_gmm_state(after_gmm, n_params=n_params, n_samples=n_samples),
        "history": history,
        "plot_path": None if plot_path is None else Path(plot_path),
        "loss": loss_name,
        "n_components": n_components,
        "cluster_method": cluster_method,
        "dpgmm_max_clusters": dpgmm_max_clusters,
        "gmm_retries": gmm_retries,
        "coverage_threshold": coverage_threshold,
    }


def _gmm_state_is_finite(gmm_state):
    return all(
        bool(torch.isfinite(gmm_state[key]).all().detach().cpu().item())
        for key in ("weights", "means", "variances", "nll")
    )


def _cosine_distance(before, after):
    '''
    cosine similarity = (A . B) / (||A|| * ||B||)
    cosine distance = 1 - cosine similarity
    '''
    before = before.detach().flatten().float()
    after = after.detach().flatten().float()
    denom = before.norm() * after.norm()
    if float(denom.detach().cpu()) <= 0.0:
        return float("nan")
    cosine_similarity = torch.dot(before, after) / denom.clamp_min(1e-12)
    return float((1.0 - cosine_similarity).detach().cpu())


def _fit_gmm(cv,n_components,seed=None, cluster_method="gmm", dpgmm_max_clusters=100, dpgmm_iterations=100, gmm_retries=10, covariance_type="diag"):
    cluster_method = str(cluster_method).strip().lower()
    if cluster_method == "gmm":
        for attempt in range(gmm_retries + 1):
            attempt_seed = None if seed is None else seed + attempt
            gmm_state = _fit_torchgmm(cv=cv, n_components=n_components, seed=attempt_seed, covariance_type=covariance_type)
            if _gmm_state_is_finite(gmm_state):
                break
        return gmm_state
    # else assume dpgmm
    return _fit_dpgmm(cv=cv,max_clusters_num=dpgmm_max_clusters,iterations_num=dpgmm_iterations,seed=seed)


def _fit_torchgmm(cv, n_components, seed=None, covariance_type="diag"):
    n_features = cv.shape[1]

    if seed is not None:
        torch.manual_seed(seed)

    device = cv.device
    estimator = tGMM(
        num_components=n_components,
        covariance_type=covariance_type,
        trainer_params = dict(
            num_nodes = 1,
            max_epochs = 100,
            accelerator = device.type,
            devices = [device.index],
            enable_progress_bar = False 
        )
    )
    fit_data = cv.detach()
    estimator.fit(fit_data)

    model = estimator.model_
    weights = model.component_probs.reshape(n_components)
    means = model.means.reshape(n_components, n_features)
    variances = model.covariances.reshape(n_components, n_features)
    log_prob = _gmm_component_log_prob(
        data=fit_data,
        weights=weights.to(device=cv.device, dtype=cv.dtype),
        means=means.to(device=cv.device, dtype=cv.dtype),
        variances=variances.to(device=cv.device, dtype=cv.dtype),
    )
    assignments = estimator.predict(fit_data)
    sample_nll = estimator.score_samples(fit_data)
    cluster_counts = torch.bincount(assignments, minlength=n_components).to(device=cv.device)

    return {
        "weights": weights.to(device=cv.device, dtype=cv.dtype),
        "means": means.to(device=cv.device, dtype=cv.dtype),
        "variances": variances.to(device=cv.device, dtype=cv.dtype),
        "assignments": assignments.to(device=cv.device),
        "cluster_counts": cluster_counts,
        "cluster_marginal_profile": _cluster_marginal_profile_from_log_prob(log_prob),
        "max_assignment_probabilities": _max_membership_probabilities_from_log_prob(log_prob),
        "nll": sample_nll.sum().to(device=cv.device, dtype=cv.dtype),
    }


def _fit_dpgmm(cv, max_clusters_num, iterations_num, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    sampler = DPGMMDiagSampler(
        init_strategy="init_data_stats",
        max_clusters_num=int(max_clusters_num),
        batch_size=2**4,
    )
    sampler = sampler.to(cv.device)
    result = sampler.fit(iterations_num=int(iterations_num), data=cv.detach())
    assignments = torch.as_tensor(result["cluster_assignment"], device=cv.device, dtype=torch.long)
    if assignments.ndim != 1 or assignments.shape[0] != cv.shape[0]:
        raise ValueError(
            "dpgmm returned assignments with an unexpected shape. "
            f"Expected shape=({cv.shape[0]},), got shape={tuple(assignments.shape)}."
        )

    active_ids, inverse = torch.unique(assignments, sorted=True, return_inverse=True)
    assignments = inverse
    cluster_counts = torch.bincount(assignments, minlength=active_ids.numel()).to(device=cv.device)
    weights = cluster_counts.to(dtype=cv.dtype) / cluster_counts.sum().clamp_min(1)

    cluster_params = result["cluster_params"]
    means = _select_active_cluster_params(
        cluster_params=cluster_params,
        active_ids=active_ids,
        key_candidates=("mean", "means", "mu"),
        device=cv.device,
        dtype=cv.dtype,
    )
    variances = _extract_active_diag_variances(
        cluster_params=cluster_params,
        active_ids=active_ids,
        n_features=cv.shape[1],
        device=cv.device,
        dtype=cv.dtype,
    )

    log_prob = _gmm_component_log_prob(
        data=cv.detach(),
        weights=weights,
        means=means,
        variances=variances,
    )
    sample_nll = -torch.logsumexp(log_prob, dim=1)

    return {
        "weights": weights,
        "means": means,
        "variances": variances,
        "assignments": assignments,
        "cluster_counts": cluster_counts,
        "cluster_marginal_profile": _cluster_marginal_profile_from_log_prob(log_prob),
        "max_assignment_probabilities": _max_membership_probabilities_from_log_prob(log_prob),
        "nll": sample_nll.sum().to(device=cv.device, dtype=cv.dtype),
    }


def _select_active_cluster_params(cluster_params, active_ids, key_candidates, device, dtype):
    for key in key_candidates:
        if key in cluster_params:
            values = torch.as_tensor(cluster_params[key], device=device, dtype=dtype)
            return values.index_select(0, active_ids.to(device=values.device))
    raise KeyError(
        f"Could not find any of {key_candidates} in dpgmm cluster_params."
    )


def _extract_active_diag_variances(cluster_params, active_ids, n_features, device, dtype):
    diag_var_keys = ("var", "vars", "variance", "variances", "cov_diag", "covariance_diag")
    for key in diag_var_keys:
        if key in cluster_params:
            values = torch.as_tensor(cluster_params[key], device=device, dtype=dtype)
            values = values.index_select(0, active_ids.to(device=values.device))
            return _coerce_diag_variances(values, n_features=n_features)

    if "cov_chol" in cluster_params:
        chol = torch.as_tensor(cluster_params["cov_chol"], device=device, dtype=dtype)
        chol = chol.index_select(0, active_ids.to(device=chol.device))
        if chol.ndim == 3:
            return chol.diagonal(dim1=-2, dim2=-1).pow(2).clamp_min(1e-12)
        if chol.ndim == 2:
            return chol.pow(2).clamp_min(1e-12)

    if "covariance" in cluster_params:
        covariance = torch.as_tensor(cluster_params["covariance"], device=device, dtype=dtype)
        covariance = covariance.index_select(0, active_ids.to(device=covariance.device))
        if covariance.ndim == 3:
            return covariance.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
        if covariance.ndim == 2:
            return covariance.clamp_min(1e-12)

def _coerce_diag_variances(values, n_features):
    if values.ndim == 1:
        values = values.unsqueeze(1)
    elif values.ndim == 3:
        values = values.diagonal(dim1=-2, dim2=-1)
    return values.clamp_min(1e-12)


def _evaluate_gmm(data, gmm_state, assignments=None):
    log_prob = _gmm_component_log_prob(
        data=data,
        weights=gmm_state["weights"],
        means=gmm_state["means"],
        variances=gmm_state["variances"],
    )
    log_norm = torch.logsumexp(log_prob, dim=1)
    if assignments is None:
        assignments = log_prob.argmax(dim=1)
    return {
        "weights": gmm_state["weights"],
        "means": gmm_state["means"],
        "variances": gmm_state["variances"],
        "assignments": assignments,
        "cluster_counts": torch.bincount(assignments, minlength=gmm_state["weights"].shape[0]).to(device=data.device),
        "cluster_marginal_profile": _cluster_marginal_profile_from_log_prob(log_prob),
        "max_assignment_probabilities": _max_membership_probabilities_from_log_prob(log_prob),
        "nll": -log_norm.sum(),
    }


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


def _gmm_num_parameters(n_components, n_features):
    weight_params = n_components - 1
    mean_params = n_components * n_features
    covariance_params = n_components * n_features
    return weight_params + mean_params + covariance_params


def _bic_from_nll(nll, n_params, n_samples):
    return 2.0 * nll + n_params * log(max(2, n_samples))


def _snapshot_gmm_state(gmm_state, n_params, n_samples):
    snapshot = {
        "weights": gmm_state["weights"].detach().clone(),
        "means": gmm_state["means"].detach().clone(),
        "variances": gmm_state["variances"].detach().clone(),
        "assignments": gmm_state["assignments"].detach().clone(),
        "cluster_counts": gmm_state["cluster_counts"].detach().clone(),
        "cluster_marginal_profile": gmm_state["cluster_marginal_profile"].detach().clone(),
        "max_assignment_probabilities": gmm_state["max_assignment_probabilities"].detach().clone(),
        "nll": gmm_state["nll"].detach().clone(),
    }
    if n_params is not None:
        snapshot["bic"] = _bic_from_nll(gmm_state["nll"], n_params, n_samples).detach().clone()
        snapshot["n_params"] = int(n_params)
    return snapshot


def _get_labels_from_dataset(datasets, loader, label_key, device):
    if loader in getattr(datasets, "_dss", {}):
        dss = datasets._dss[loader]
    else:
        dss = datasets._dss_ori[loader]

    labels = dss[label_key]
    return torch.as_tensor(labels, device=device)


def _max_membership_probabilities_from_log_prob(log_prob):
    log_norm = torch.logsumexp(log_prob, dim=1, keepdim=True)
    posterior = torch.exp(log_prob - log_norm)
    return posterior.max(dim=1).values


def _cluster_marginal_profile_from_log_prob(log_prob):
    log_norm = torch.logsumexp(log_prob, dim=1, keepdim=True)
    posterior = torch.exp(log_prob - log_norm)
    return posterior.mean(dim=0)


def _compute_clustering_metrics(projected,gmm_state,loss_name,n_params,n_samples,seed=None,labels=None,
                                coverage_threshold=0.9, n_classes=None):
    assignments = gmm_state["assignments"]
    counts = gmm_state.get("cluster_counts")
    if counts is None:
        counts = torch.bincount(assignments, minlength=gmm_state["weights"].shape[0]).to(projected.device)
    counts = counts.to(projected.dtype)
    active_clusters = int((counts > 0).sum().item())
    probs = counts / counts.sum().clamp_min(1.0)
    nonzero = probs > 0
    cluster_entropy = -(probs[nonzero] * probs[nonzero].log()).sum()
    max_entropy = log(max(2, gmm_state["weights"].shape[0]))
    normalized_entropy = cluster_entropy / max_entropy
    silhouette = _silhouette_score(projected, assignments, seed=seed)
    nll = gmm_state["nll"]
    bic = None if n_params is None else _bic_from_nll(nll, n_params, n_samples)
    if loss_name == "nll":
        objective = nll
    elif loss_name == "silhouette":
        objective = silhouette
    else:
        objective = bic


    metrics = {
        "objective": float(objective.detach().cpu()),
        "nll": float(nll.detach().cpu()),
        "mean_nll": float((nll / n_samples).detach().cpu()),
        "bic": None if bic is None else float(bic.detach().cpu()),
        "complexity": None if n_params is None else int(n_params),
        "bic_penalty": None if n_params is None else float((n_params * log(max(2, n_samples)))),
        "active_clusters": active_clusters,
        "normalized_cluster_entropy": float(normalized_entropy.detach().cpu()),
        "silhouette": float(silhouette.detach().cpu()),
    }

    if labels is not None:
        coverage_metrics = _coverage(
            assignments=assignments,
            labels=labels,
            n_clusters=gmm_state["weights"].shape[0],
            coverage_threshold=coverage_threshold,
            n_classes=n_classes,
            dtype=projected.dtype,
        )
        metrics.update(coverage_metrics)

    return metrics


def _coverage(assignments, labels, n_clusters, coverage_threshold, n_classes=None, dtype=torch.float32):
    labels = torch.as_tensor(labels, device=assignments.device)
    assignments = assignments.long()

    labels, n_classes = _labels_to_class_ids(
        labels=labels,
        n_classes=n_classes,
        device=assignments.device,
    )
    empp = _compute_empp_from_class_ids(
        assignments=assignments,
        labels=labels,
        n_clusters=n_clusters,
        n_classes=n_classes,
        dtype=dtype
    )

    class_represented = (empp.transpose(0, 1) >= coverage_threshold).any(dim=1)
    cluster_represented = (empp >= coverage_threshold).any(dim=1)

    return {
        "class_coverage": float(class_represented.to(dtype).mean().detach().cpu()),
        "cluster_coverage": float(cluster_represented.to(dtype).mean().detach().cpu()),
    }

def _labels_to_class_ids(labels, n_classes=None, device=None):
    labels = torch.as_tensor(labels, device=device)
    if n_classes is None:
        n_classes = int(labels.max().item()) + 1
    return labels.long(), int(n_classes)

def _compute_empp_from_class_ids(assignments, labels, n_clusters, n_classes, dtype=torch.float32):
    flat_idx = assignments * int(n_classes) + labels
    joint_counts = torch.bincount(flat_idx, minlength=int(n_clusters) * int(n_classes)).to(dtype)
    empp = joint_counts.reshape(int(n_clusters), int(n_classes))
    counts = torch.bincount(assignments, minlength=int(n_clusters)).unsqueeze(1).to(dtype)
    empp = empp / counts.clamp_min(1.0)
    return empp

def _save_clustering_stats_plot(**kwargs):
    before_metrics = kwargs["before_metrics"]
    after_metrics = kwargs["after_metrics"]
    plot_path = Path(kwargs["plot_path"])
    loss_name = kwargs["loss_name"]
    layer_name = kwargs.get("layer_name", "")
    summary_text = kwargs.get("summary_text")
    before_gmm = kwargs.get("before_gmm")
    after_gmm = kwargs.get("after_gmm")
    cluster_population_top_k = int(kwargs.get("cluster_population_top_k", 10))
    cluster_population_mode = kwargs.get("cluster_population_mode", "both")
    projection_cosine_distance = kwargs.get("projection_cosine_distance")
    improvement_title = kwargs.get("improvement_title", "Before vs after improvement")
    marginal_profile_title = kwargs.get(
        "marginal_profile_title",
        "Cluster marginal posterior profile",
    )
    population_title = kwargs.get("population_title", "Cluster populations")
    before_population_title = kwargs.get("before_population_title", "Before cluster populations")
    after_population_title = kwargs.get("after_population_title", "After cluster populations")

    rows = _metric_rows(before_metrics, after_metrics)
    table_data = []
    for key, label in rows:
        before = before_metrics[key]
        after = after_metrics[key]
        delta = None if before is None or after is None else after - before
        table_data.append([label, _format(before), _format(after), _format(delta)])
    if projection_cosine_distance is not None:
        table_data.append([
            "V^T cos. distance",
            "-",
            "-",
            _format(projection_cosine_distance),
        ])

    if cluster_population_mode == "single":
        fig = plt.figure(figsize=(18, 12))
        grid = fig.add_gridspec(
            3,
            2,
            width_ratios=[1.35, 1.0],
            height_ratios=[1.0, 0.9, 1.0],
        )
        axes = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :]),
            fig.add_subplot(grid[2, :]),
        ]
    else:
        fig = plt.figure(figsize=(18, 14))
        grid = fig.add_gridspec(3,2,
            width_ratios=[1.35, 1.0],
            height_ratios=[1.0, 0.9, 1.1],
        )
        axes = [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :]),
            fig.add_subplot(grid[2, 0]),
            fig.add_subplot(grid[2, 1]),
        ]

    axes[0].axis("off")
    table = axes[0].table(
        cellText=table_data,
        colLabels=["Metric", "Before", "After", "Delta"],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)
    axes[0].set_title(f"Clustering quality summary ({loss_name.upper()} optimization)")
    if summary_text:
        axes[0].text(0.5, 0.04,
            summary_text,
            ha="center", va="bottom",
            transform=axes[0].transAxes,
            fontsize=10)
        
    fig.suptitle(f"Layer: {layer_name}", fontsize=14)

    objective_lower_is_better = loss_name != "silhouette"
    improvement_rows = [
        ("Objective", _safe_relative_improvement(before_metrics["objective"], after_metrics["objective"], lower_is_better=objective_lower_is_better)),
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

    axes[1].axvline(0.0, color="black", linewidth=1.0)
    colors = ["tab:green" if value >= 0 else "tab:red" for value in values]
    axes[1].barh(labels, values, color=colors)
    axes[1].set_xlabel("Relative improvement")
    axes[1].set_title(improvement_title)
    axes[1].set_xlim(
        min(-0.05, min(values + [0.0]) * 1.1 if values else -0.05),
        max(0.05, max(values + [0.0]) * 1.1 if values else 0.05),
    )

    for idx, value in enumerate(values):
        axes[1].text(value, idx, f" {value * 100:.2f}%", va="center")

    _render_cluster_marginal_likelihood_panel(
        ax=axes[2],
        title=marginal_profile_title,
        before_gmm=before_gmm,
        after_gmm=after_gmm)

    if cluster_population_mode == "single":
        axes[3].axis("off")
        _render_cluster_population_panel(
            ax=axes[3],
            title=population_title,
            gmm_state=before_gmm if before_gmm is not None else after_gmm,
            top_k=cluster_population_top_k,
        )
    else:
        axes[3].axis("off")
        axes[4].axis("off")
        _render_cluster_population_panel(
            ax=axes[3],
            title=before_population_title,
            gmm_state=before_gmm,
            top_k=cluster_population_top_k,
        )
        _render_cluster_population_panel(
            ax=axes[4],
            title=after_population_title,
            gmm_state=after_gmm,
            top_k=cluster_population_top_k,
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97) if layer_name else None)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_cluster_population_panel(ax, title, gmm_state, top_k):
    ax.set_title(title)
    ax.axis("off")

    if gmm_state is None:
        ax.text(0.01, 0.99, "No GMM state available.", va="top", ha="left")
        return

    summary = _cluster_population_summary(gmm_state=gmm_state, top_k=top_k)
    ax.text(0.01,0.99,summary,va="top",ha="left",family="monospace",fontsize=10)


def _render_cluster_marginal_likelihood_panel(ax,title,before_gmm,after_gmm):
    ax.set_title(title)

    before_profile = _sorted_cluster_marginal_posteriors(before_gmm)
    after_profile = _sorted_cluster_marginal_posteriors(after_gmm)

    if before_profile is None and after_profile is None:
        ax.text(0.5, 0.5, "No cluster weights available.", ha="center", va="center")
        ax.set_axis_off()
        return

    if before_profile is not None:
        x_before = list(range(1, len(before_profile) + 1))
        ax.plot(
            x_before,
            before_profile,
            label="Before",
            color="tab:blue",
            linewidth=2.0,
            marker="o",
            markersize=3.0,
        )

    if after_profile is not None:
        x_after = list(range(1, len(after_profile) + 1))
        ax.plot(
            x_after,
            after_profile,
            label="After",
            color="tab:orange",
            linewidth=2.0,
            marker="o",
            markersize=3.0,
        )

    max_clusters = max(
        0 if before_profile is None else len(before_profile),
        0 if after_profile is None else len(after_profile),
    )
    if max_clusters > 0:
        if max_clusters == 1:
            ax.set_xlim(0.5, 1.5)
            ax.set_xticks([1])
        else:
            ax.set_xlim(1, max_clusters)
            if max_clusters <= 20:
                ax.set_xticks(list(range(1, max_clusters + 1)))

    ax.set_xlabel("Clusters ordered by descending marginal posterior")
    ax.set_ylabel("phi_k = P(z=k | theta)")
    ax.grid(True, axis="both", alpha=0.25)
    if before_profile is not None or after_profile is not None:
        ax.legend()


def _sorted_cluster_marginal_posteriors(gmm_state):
    if gmm_state is None:
        return None

    weights = gmm_state.get("weights")
    if weights is None:
        return None

    sorted_profile = torch.sort(torch.as_tensor(weights).detach().float().cpu(), descending=True).values
    return sorted_profile.tolist()


def _cluster_population_summary(gmm_state, top_k=10):
    n_clusters = int(gmm_state["weights"].shape[0])
    counts = gmm_state.get("cluster_counts")
    if counts is None:
        assignments = gmm_state["assignments"].long()
        counts = torch.bincount(assignments, minlength=n_clusters)
    counts = counts.detach().cpu().tolist()
    indexed_counts = list(enumerate(counts))

    most_populated = sorted(indexed_counts, key=lambda item: (-item[1], item[0]))[:top_k]
    least_populated = sorted(indexed_counts, key=lambda item: (item[1], item[0]))[:top_k]
    active_clusters = sum(count > 0 for _, count in indexed_counts)
    total_samples = sum(counts)

    lines = [
        f"Total samples: {total_samples}",
        f"Active clusters: {active_clusters}/{n_clusters}",
        "",
        f"Top {min(top_k, n_clusters)} most populated",
    ]
    lines.extend(_format_cluster_count_lines(most_populated))
    lines.extend([
        "",
        f"Top {min(top_k, n_clusters)} least populated",
    ])
    lines.extend(_format_cluster_count_lines(least_populated))
    return "\n".join(lines)


def _format_cluster_count_lines(indexed_counts):
    return [f"cluster {idx:>4}: {count}" for idx, count in indexed_counts]


def _metric_rows(before_metrics, after_metrics):
    candidate_rows = [
        ("objective", "Objective"),
        ("nll", "NLL"),
        ("mean_nll", "Mean NLL"),
        ("bic", "BIC"),
        ("complexity", "Complexity (p)"),
        ("bic_penalty", "BIC penalty"),
        ("active_clusters", "Active clusters"),
        ("normalized_cluster_entropy", "Norm. entropy"),
        ("silhouette", "Silhouette"),
        ("class_coverage", "Class coverage"),
        ("cluster_coverage", "Cluster coverage"),
    ]
    rows = []
    for key, label in candidate_rows:
        if key in before_metrics and key in after_metrics:
            if before_metrics[key] is None and after_metrics[key] is None:
                continue
            rows.append((key, label))
    return rows


def _silhouette_score(data, assignments, seed=None):
    del seed
    unique_clusters, inverse = torch.unique(assignments, sorted=True, return_inverse=True)
    if unique_clusters.numel() < 2:
        return torch.tensor(float("nan"), device=data.device, dtype=data.dtype)

    distances = torch.cdist(data, data, p=2)
    memberships = torch.nn.functional.one_hot(inverse, num_classes=unique_clusters.numel()).to(data.dtype)
    cluster_sizes = memberships.sum(dim=0)
    distance_sums = distances @ memberships

    own_cluster_sizes = cluster_sizes[inverse]
    own_distance_sums = distance_sums.gather(1, inverse.unsqueeze(1)).squeeze(1)
    a_i = torch.zeros(data.shape[0], device=data.device, dtype=data.dtype)
    valid_intra = own_cluster_sizes > 1
    a_i[valid_intra] = own_distance_sums[valid_intra] / (own_cluster_sizes[valid_intra] - 1)

    mean_cluster_distances = distance_sums / cluster_sizes.clamp_min(1).unsqueeze(0)
    mean_cluster_distances.scatter_(1, inverse.unsqueeze(1), float("inf"))
    b_i = mean_cluster_distances.min(dim=1).values

    silhouette_values = torch.zeros(data.shape[0], device=data.device, dtype=data.dtype)
    valid_inter = torch.isfinite(b_i)
    valid = valid_intra & valid_inter
    denom = torch.maximum(a_i[valid], b_i[valid]).clamp_min(1e-12)
    silhouette_values[valid] = (b_i[valid] - a_i[valid]) / denom

    return silhouette_values.mean()


def _format(value):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if value != value:
        return "nan"
    return f"{value:.6f}"


def _safe_relative_improvement(before, after, lower_is_better):
    if before is None or after is None:
        return None
    if before != before or after != after:
        return None
    if abs(before) < 1e-12:
        return None
    if lower_is_better:
        return (before - after) / abs(before)
    return (after - before) / max(abs(before), 1e-12)
