import torch
import os
import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def localization_from_conceptogram(**kwargs):
    """
    Computes localization for a conceptogram matrix M:

        L = sum_{i,j} M_ij^2 / (sum_{i,j} M_ij)^2

    Args (via kwargs):
        M : torch.Tensor
            Conceptogram for one sample. Expected shape [n_layers, d] (2D).

    Returns:
        torch.Tensor (scalar): localization value
    """
    M = kwargs["M"]
    eps = kwargs.get("eps", 1e-12)

    if not torch.is_tensor(M):
        raise ValueError("M must be a torch.Tensor")
    if M.ndim != 2:
        raise ValueError(f"Expected M to be 2D [n_layers, d], got shape {tuple(M.shape)}")

    num = (M ** 2).sum()
    den = M.sum().pow(2).clamp_min(eps)
    return num / den

def binary_auc_from_scores(**kwargs):
    """
    Rank-based ROC-AUC (Mann–Whitney U) with average-rank tie handling.

    Args (via kwargs):
        scores : 1D torch.Tensor (float)
        labels : 1D torch.Tensor (0/1 or bool), 1 = positive

    Returns:
        float (AUC) or NaN if undefined
    """
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


def plot_distribution(**kwargs):
    """
    Plot correct vs incorrect localization distributions on the same figure:
      - correct in green
      - incorrect in red
    Also prints AUC and FPR95 (correct=positive).

    Args (via kwargs):
        Ls : 1D torch.Tensor
            Per-sample localization values for ALL samples (aligned with ds results order).
        results : 1D torch.Tensor (bool or 0/1)
            True for correct predictions, False for incorrect.
        save_dir : str or Path
        file_name : str (optional)
        title : str (optional)
        bins : int (optional)
        verbose : bool (optional)
    """
    Ls = kwargs["Ls"].detach().cpu().flatten()
    results = kwargs["results"].detach().cpu().bool().flatten()

    save_dir = kwargs["save_dir"]
    file_name = kwargs.get("file_name", "localization_distribution_overlay.png")
    title = kwargs.get("title", "Localization distribution (correct vs incorrect)")
    bins = kwargs.get("bins", 50)
    verbose = kwargs.get("verbose", True)

    os.makedirs(save_dir, exist_ok=True)

    if Ls.numel() != results.numel():
        raise ValueError(f"Ls has {Ls.numel()} elements but results has {results.numel()} elements.")

    s_oks = Ls[results]
    s_kos = Ls[~results]

    # AUC
    auc = binary_auc_from_scores(scores=Ls, labels=results.int())

    # FPR@95
    if s_oks.numel() == 0 or s_kos.numel() == 0:
        fpr95 = float("nan")
        threshold = float("nan")
    else:
        sorted_pos, _ = torch.sort(s_oks, descending=True)
        tpr95_index = int(torch.ceil(torch.tensor(0.95 * sorted_pos.numel())).item()) - 1
        threshold = sorted_pos[tpr95_index].item()
        fpr95 = (s_kos >= threshold).float().mean().item()

    # Plot overlay (green correct, red incorrect)
    plt.figure()
    plt.hist(s_oks.numpy(), bins=bins, density=True, alpha=0.55, color="green", label=f"correct (n={s_oks.numel()})")
    plt.hist(s_kos.numpy(), bins=bins, density=True, alpha=0.55, color="red", label=f"incorrect (n={s_kos.numel()})")
    plt.xlabel("Localization")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()

    if verbose:
        if auc == auc:
            print(f"AUC: {auc:.4f}")
        else:
            print("AUC: NaN (need both correct and incorrect samples)")

        if fpr95 == fpr95:
            print(f"FPR95: {fpr95:.4f} (threshold@TPR95={threshold:.6f})")
        else:
            print("FPR95: NaN (need both correct and incorrect samples)")

    return {"auc": auc, "fpr95": fpr95, "threshold_tpr95": threshold}


@torch.no_grad()
def localization_from_peepholes(**kwargs):
    """
    Computes localization for all samples in ds_key (auto-extracted from ds),
    and (optionally) plots correct vs incorrect distributions.

    Returns:
        dict with:
            L_avg : float
            Ls : 1D tensor (all samples)
            auc : float
            fpr95 : float
            threshold_tpr95 : float
    """
    phs = kwargs["phs"]
    ds = kwargs["ds"]
    ds_key = kwargs["ds_key"]
    target_modules = kwargs["target_modules"]

    eps = kwargs.get("eps", 1e-12)
    plot = kwargs.get("plot", False)
    verbose = kwargs.get("verbose", True)

    results = ds._dss[ds_key]["result"].detach().cpu().bool()
    n = results.shape[0]
    all_samples = list(range(n))

    Ls = []
    for sample_idx in all_samples:
        M = torch.stack(
            [phs._phs[ds_key][m][sample_idx] for m in target_modules],
            dim=0
        )
        L = localization_from_conceptogram(M=M, eps=eps)
        Ls.append(L)

    Ls = torch.stack(Ls).detach().cpu().flatten()
    L_avg = float(Ls.mean().item())

    # --- compute metrics ALWAYS (not only when plotting) ---
    auc = binary_auc_from_scores(scores=Ls, labels=results.int())

    s_oks = Ls[results]
    s_kos = Ls[~results]
    if s_oks.numel() == 0 or s_kos.numel() == 0:
        fpr95 = float("nan")
        threshold = float("nan")
    else:
        sorted_pos, _ = torch.sort(s_oks, descending=True)
        tpr95_index = int(torch.ceil(torch.tensor(0.95 * sorted_pos.numel())).item()) - 1
        threshold = sorted_pos[tpr95_index].item()
        fpr95 = (s_kos >= threshold).float().mean().item()

    if verbose:
        print(f"Average localization L={L_avg:.6f} over {n} samples in ds_key={ds_key}")
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
        file_name = kwargs.get("file_name", "localization_distribution_overlay.png")
        title = kwargs.get("title", f"Localization distribution ({ds_key})")
        bins = kwargs.get("bins", 50)

        # plot overlay only; do NOT re-compute metrics
        os.makedirs(save_dir, exist_ok=True)
        plt.figure()
        plt.hist(s_oks.numpy(), bins=bins, density=True, alpha=0.55, color="green",
                 label=f"correct (n={s_oks.numel()})")
        plt.hist(s_kos.numpy(), bins=bins, density=True, alpha=0.55, color="red",
                 label=f"incorrect (n={s_kos.numel()})")
        plt.xlabel("Localization")
        plt.ylabel("Density")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()

    return {
        "L_avg": L_avg,
        "Ls": Ls,
        "auc": float(auc),
        "fpr95": float(fpr95),
        "threshold_tpr95": float(threshold),
    }

def localization_means(**kwargs):
    """
    Compute average localization for:
      - all samples
      - correct samples only
      - incorrect samples only

    Args (via kwargs):
        Ls : 1D torch.Tensor
            Localization per sample (length n).
        results : 1D torch.Tensor (bool or 0/1)
            True for correct, False for incorrect (length n).

    Returns:
        dict with float means.
    """
    Ls = kwargs["Ls"].detach().cpu().flatten()
    results = kwargs["results"].detach().cpu().bool().flatten()

    if Ls.numel() != results.numel():
        raise ValueError(f"Ls has {Ls.numel()} elements but results has {results.numel()} elements.")

    mean_all = float(Ls.mean().item())

    if results.any():
        mean_correct = float(Ls[results].mean().item())
    else:
        mean_correct = float("nan")

    if (~results).any():
        mean_incorrect = float(Ls[~results].mean().item())
    else:
        mean_incorrect = float("nan")

    return {
        "mean_all": mean_all,
        "mean_correct": mean_correct,
        "mean_incorrect": mean_incorrect,
        "n_all": int(Ls.numel()),
        "n_correct": int(results.sum().item()),
        "n_incorrect": int((~results).sum().item()),
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
def localization_pmax_correlations(**kwargs):
    """
    Compute correlation between localization and pmax (max softmax prob) for a fixed set of layers.
    Args (via kwargs):
        phs : Peepholes instance
        ds : ParsedDataset
        ds_key : str (e.g., "CIFAR100-test")
        target_modules : list[str] (the layers you want to use)
        eps : float (optional, passed to localization_from_peepholes)
        verbose : bool (optional)

    Returns:
        dict with Pearson/Spearman correlations:
          - overall
          - correct-only
          - incorrect-only
        plus counts.
    """
    phs = kwargs["phs"]
    ds = kwargs["ds"]
    ds_key = kwargs["ds_key"]
    target_modules = kwargs["target_modules"]

    # localization
    out = localization_from_peepholes(phs=phs,
        ds=ds, ds_key=ds_key,
        target_modules=target_modules,
    )
    Ls = out["Ls"].detach().cpu().flatten()

    # pmax from probabilities
    output = ds._dss[ds_key]["output"].detach().cpu()   # logits/scores
    pmax = torch.softmax(output, dim=1).max(dim=1).values.flatten()

    results = ds._dss[ds_key]["result"].detach().cpu().bool().flatten()

    if Ls.numel() != pmax.numel():
        raise ValueError(f"Mismatch: Ls has {Ls.numel()} elems but pmax has {pmax.numel()} elems")

    # plot conf vs localization
    save_dir = kwargs.get("save_dir", ".")
    file_name = kwargs.get("file_name", f"confidence_vs_localization_{ds_key}.png")
    title = kwargs.get("title", f"Confidence vs Localization ({ds_key})")

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))

    plt.scatter(pmax[~results].numpy(), Ls[~results].numpy(),
        s=8, alpha=0.4,
        color="red",
        label=f"Incorrect (n={(~results).sum().item()})",
    )

    plt.scatter(pmax[results].numpy(), Ls[results].numpy(),
        s=8, alpha=0.4,
        color="green",
        label=f"Correct (n={results.sum().item()})",
    )

    plt.xlabel("Max softmax probability (confidence)")
    plt.ylabel("Localization")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()

    out_corr = {}
    out_corr["n_all"] = int(Ls.numel())
    out_corr["n_correct"] = int(results.sum().item())
    out_corr["n_incorrect"] = int((~results).sum().item())

    # overall
    out_corr["pearson_all"] = pearson_corr(Ls, pmax)
    out_corr["spearman_all"] = spearman_corr(Ls, pmax)

    # correct-only
    if results.any():
        out_corr["pearson_correct"] = pearson_corr(Ls[results], pmax[results])
        out_corr["spearman_correct"] = spearman_corr(Ls[results], pmax[results])
    else:
        out_corr["pearson_correct"] = float("nan")
        out_corr["spearman_correct"] = float("nan")

    # incorrect-only
    if (~results).any():
        out_corr["pearson_incorrect"] = pearson_corr(Ls[~results], pmax[~results])
        out_corr["spearman_incorrect"] = spearman_corr(Ls[~results], pmax[~results])
    else:
        out_corr["pearson_incorrect"] = float("nan")
        out_corr["spearman_incorrect"] = float("nan")

    print(f"Computed correlations for ds_key={ds_key} with {len(target_modules)} layers")
    print(
        f"Pearson(all)={out_corr['pearson_all']:.4f}, Spearman(all)={out_corr['spearman_all']:.4f} | "
        f"n_correct={out_corr['n_correct']}, n_incorrect={out_corr['n_incorrect']}"
    )


    return out_corr
