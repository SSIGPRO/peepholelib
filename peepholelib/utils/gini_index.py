import os
from matplotlib import pyplot as plt
import torch

@torch.no_grad()
@torch.no_grad()
def gini_from_conceptogram(**kwargs):
    """
    Batched Gini for nonnegative tensors.

    Args (via kwargs):
        X : torch.Tensor, shape [n_samples, n_layers, d] OR [n_samples, D]
            Nonnegative entries.
        eps : float (optional)

    Returns:
        torch.Tensor, shape [n_samples] (float64): Gini in [0,1]
    """
    X = kwargs["X"]
    eps = kwargs.get("eps", 1e-12)

    if not torch.is_tensor(X):
        raise ValueError("X must be a torch.Tensor")

    X = X.to(torch.float64)

    if X.ndim == 3:
        x = X.flatten(1)          # [n_samples, D]
    elif X.ndim == 2:
        x = X
    S = x.sum(dim=1)            
    n = x.shape[1]

    x_sorted, _ = torch.sort(x, dim=1)  # ascending, [n_samples, n]

    k = torch.arange(1, n + 1, device=x_sorted.device, dtype=torch.float64).view(1, -1)

    num = (k * x_sorted).sum(dim=1)     
    den = (n * S).clamp_min(eps)

    g = (2.0 * num) / den - (n + 1.0) / n
    g = torch.where(S <= eps, torch.zeros_like(g), g)

    return g.clamp(0.0, 1.0)


def binary_auc_from_scores(**kwargs):

    scores = kwargs["scores"].detach().cpu().flatten()
    labels = kwargs["labels"].detach().cpu().flatten().to(torch.int64)

    if scores.numel() == 0 or labels.numel() == 0:
        return float("nan")
    if labels.unique().numel() < 2:
        return float("nan")

    order = torch.argsort(scores, stable=True)
    s = scores[order]
    y = labels[order]

    n = s.numel()
    ranks = torch.arange(1, n + 1, dtype=torch.float64)

    diff = torch.ones(n, dtype=torch.bool)
    diff[1:] = s[1:] != s[:-1]
    starts = torch.nonzero(diff, as_tuple=False).flatten()

    ends = torch.empty_like(starts)
    ends[:-1] = starts[1:] - 1
    ends[-1] = n - 1

    ranks_tied = ranks.clone()
    for a, b in zip(starts.tolist(), ends.tolist()):
        if b > a:
            avg_rank = (ranks[a] + ranks[b]) / 2.0
            ranks_tied[a:b + 1] = avg_rank

    pos = (y == 1)
    n_pos = int(pos.sum().item())
    n_neg = int(n - n_pos)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_ranks_pos = ranks_tied[pos].sum().item()
    u = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    auc = u / (n_pos * n_neg)
    return float(auc)

@torch.no_grad()
def gini_from_peepholes(**kwargs):
    """
    Returns:
        dict with:
            g_avg, gs, auc, fpr95, threshold_tpr95
    """
    phs = kwargs["phs"]
    ds = kwargs["ds"]
    ds_key = kwargs["ds_key"]
    target_modules = kwargs["target_modules"]

    device = kwargs.get("device", None)
    eps = kwargs.get("eps", 1e-12)
    plot = kwargs.get("plot", False)
    verbose = kwargs.get("verbose", True)

    results = ds._dss[ds_key]["result"].detach().cpu().bool().flatten()
    n_samples = results.numel()

    # Build [n_layers, n_samples, d] then permute to [n_samples, n_layers, d]
    # (keeps the Python loop over layers, but avoids looping over samples)
    layer_tensors = []
    for m in target_modules:
        t = phs._phs[ds_key][m]  # expected [n_samples, d] or list-like of tensors
        if not torch.is_tensor(t):
            t = torch.stack(list(t), dim=0)
        layer_tensors.append(t)

    X = torch.stack(layer_tensors, dim=0).permute(1, 0, 2)  # [n_samples, n_layers, d]

    if device is not None:
        X = X.to(device, non_blocking=True)

    gs = gini_from_conceptogram(X=X, eps=eps) 
    g_avg = float(gs.mean().detach().cpu().item())

    gs_cpu = gs.detach().cpu().flatten()

    auc = binary_auc_from_scores(scores=gs_cpu, labels=results.int())

    s_oks = gs_cpu[results]
    s_kos = gs_cpu[~results]
    if s_oks.numel() == 0 or s_kos.numel() == 0:
        fpr95 = float("nan")
        threshold = float("nan")
    else:
        sorted_pos, _ = torch.sort(s_oks, descending=True)
        tpr95_index = int(torch.ceil(torch.tensor(0.95 * sorted_pos.numel())).item()) - 1
        threshold = sorted_pos[tpr95_index].item()
        fpr95 = (s_kos >= threshold).float().mean().item()

    if verbose:
        print(f"Average Gini g={g_avg:.6f} over {n_samples} samples in ds_key={ds_key}")
        if auc == auc:
            print(f"AUC: {auc:.4f}")
        else:
            print("AUC: NaN (need both correct and incorrect samples)")
        if fpr95 == fpr95:
            print(f"FPR95: {fpr95:.4f} (threshold@TPR95={threshold:.6f})")
        else:
            print("FPR95: NaN (need both correct and incorrect samples)")

    if plot:
        save_dir = kwargs["save_dir"]
        file_name = kwargs.get("file_name", "gini_distribution_overlay.png")
        title = kwargs.get("title", f"Gini distribution ({ds_key})")
        bins = kwargs.get("bins", 50)

        os.makedirs(save_dir, exist_ok=True)
        plt.figure()
        plt.hist(s_oks.numpy(), bins=bins, density=True, alpha=0.55, color="green",
                 label=f"correct (n={s_oks.numel()})")
        plt.hist(s_kos.numpy(), bins=bins, density=True, alpha=0.55, color="red",
                 label=f"incorrect (n={s_kos.numel()})")
        plt.xlabel("Gini (sparsity)")
        plt.ylabel("Density")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()

    return {
        "g_avg": g_avg,
        "gs": gs_cpu,
        "auc": float(auc),
        "fpr95": float(fpr95),
        "threshold_tpr95": float(threshold),
    }

def _rankdata_average_ties(v):
    v = v.detach().cpu().flatten()
    order = torch.argsort(v, stable=True)
    s = v[order]

    n = s.numel()
    ranks = torch.arange(1, n + 1, dtype=torch.float64)

    diff = torch.ones(n, dtype=torch.bool)
    diff[1:] = s[1:] != s[:-1]
    starts = torch.nonzero(diff, as_tuple=False).flatten()

    ends = torch.empty_like(starts)
    ends[:-1] = starts[1:] - 1
    ends[-1] = n - 1

    ranks_tied = ranks.clone()
    for a, b in zip(starts.tolist(), ends.tolist()):
        if b > a:
            avg_rank = (ranks[a] + ranks[b]) / 2.0
            ranks_tied[a:b + 1] = avg_rank

    inv = torch.empty_like(order)
    inv[order] = torch.arange(n)
    return ranks_tied[inv]


def pearson_corr(x, y, eps=1e-12):
    x = x.detach().cpu().flatten().to(torch.float64)
    y = y.detach().cpu().flatten().to(torch.float64)

    if x.numel() != y.numel():
        raise ValueError(f"x has {x.numel()} elems, y has {y.numel()} elems")

    x = x - x.mean()
    y = y - y.mean()

    num = (x * y).sum()
    den = torch.sqrt((x * x).sum().clamp_min(eps) * (y * y).sum().clamp_min(eps))
    return float((num / den).item())


def spearman_corr(x, y):
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    return pearson_corr(rx, ry)


@torch.no_grad()
def gini_pmax_correlations(**kwargs):
    """
    Compute Spearman correlation between Gini sparsity and pmax (max softmax prob)

    Returns:
        dict with Spearman correlations: overall, correct-only, incorrect-only
    """
    phs = kwargs["phs"]
    ds = kwargs["ds"]
    ds_key = kwargs["ds_key"]
    target_modules = kwargs["target_modules"]

    out = gini_from_peepholes(
        phs=phs,
        ds=ds,
        ds_key=ds_key,
        target_modules=target_modules,
        eps=kwargs.get("eps", 1e-12),
        plot=False,
        verbose=kwargs.get("verbose", True),
    )
    gs = out["gs"].detach().cpu().flatten().to(torch.float64)

    output = ds._dss[ds_key]["output"].detach().cpu()
    pmax = torch.softmax(output, dim=1).max(dim=1).values.flatten().to(torch.float64)

    results = ds._dss[ds_key]["result"].detach().cpu().bool().flatten()

    if gs.numel() != pmax.numel():
        raise ValueError(f"Mismatch: gs has {gs.numel()} elems but pmax has {pmax.numel()} elems")

    # scatter plot
    save_dir = kwargs.get("save_dir", ".")
    file_name = kwargs.get("file_name", f"confidence_vs_gini_{ds_key}.png")
    title = kwargs.get("title", f"Confidence vs Gini ({ds_key})")

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))

    plt.scatter(
        pmax[~results].numpy(), gs[~results].numpy(),
        s=8, alpha=0.4,
        color="red",
        label=f"Incorrect (n={(~results).sum().item()})",
    )
    plt.scatter(
        pmax[results].numpy(), gs[results].numpy(),
        s=8, alpha=0.4,
        color="green",
        label=f"Correct (n={results.sum().item()})",
    )

    plt.xlabel("Max softmax probability (confidence)")
    plt.ylabel("Gini (sparsity)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()

    # Spearman correlations 
    out_corr = {
        "n_all": int(gs.numel()),
        "n_correct": int(results.sum().item()),
        "n_incorrect": int((~results).sum().item()),
        "spearman_all": spearman_corr(gs, pmax),
    }

    if results.any():
        out_corr["spearman_correct"] = spearman_corr(gs[results], pmax[results])
    else:
        out_corr["spearman_correct"] = float("nan")

    if (~results).any():
        out_corr["spearman_incorrect"] = spearman_corr(gs[~results], pmax[~results])
    else:
        out_corr["spearman_incorrect"] = float("nan")

    print(f"Computed Spearman correlations for ds_key={ds_key} with {len(target_modules)} layers")
    print(
        f"Spearman(all)={out_corr['spearman_all']:.4f} | "
        f"n_correct={out_corr['n_correct']}, n_incorrect={out_corr['n_incorrect']}"
    )

    return out_corr
