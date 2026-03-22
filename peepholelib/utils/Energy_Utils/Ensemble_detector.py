"""
peepholelib.ensemble_detector
==============================
Tail-energy ensemble detector: routes inputs between a standard and a robust
model based on a single threshold on the SVD tail energy of an intermediate
layer's corevector.

Threshold calibration modes
----------------------------
  "fpr"         — fix the clean false-positive rate
  "asr"         — fix the post-ensemble ASR on a chosen calibration attack
  "clean_acc"   — fix the ensemble clean accuracy

Public API
----------
  load_td(path)
  compute_tail_energy(cv, layer, mask, tail) -> np.ndarray
  calibrate_threshold(mode, *, ...) -> (threshold, info_dict)
  evaluate_ensemble(...) -> dict
  plot_threshold(...)
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import gaussian_kde
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score
from tensordict import PersistentTensorDict


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_td(path: str | Path) -> PersistentTensorDict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")
    return PersistentTensorDict.from_h5(str(path))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_tail_energy(
    cv,
    layer: str,
    mask: torch.Tensor | np.ndarray,
    tail: int,
) -> np.ndarray:
    """Sum of squared values of the last `tail` SVD components."""
    if layer not in cv.keys():
        available = list(cv.keys())
        raise KeyError(
            f"Layer '{layer}' not found.\n"
            f"Available layers (first 10): {available[:10]}"
        )
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    z = cv[layer][:].float()[mask]
    energy = (z[:, -tail:] ** 2).sum(dim=1)
    return energy.cpu().numpy()


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def _ensemble_asr(
    thresh: float,
    adv_energy: np.ndarray,
    adv_correct_standard: np.ndarray,
    adv_correct_robust: np.ndarray,
) -> float:
    flagged = adv_energy > thresh
    correct = (~flagged & adv_correct_standard) | (flagged & adv_correct_robust)
    return (~correct).mean()


def _ensemble_clean_acc(
    thresh: float,
    clean_energy: np.ndarray,
    clean_correct_robust: np.ndarray,
    base_clean_acc: float,
) -> float:
    flagged  = clean_energy > thresh
    retained = (~flagged) | (flagged & clean_correct_robust)
    return retained.mean() * base_clean_acc


def calibrate_threshold(
    mode: str,
    *,
    clean_energy: np.ndarray,
    clean_correct_robust: np.ndarray,
    base_clean_acc: float,
    # required for mode == "fpr"
    target_fpr: float | None = None,
    # required for mode == "asr"
    target_asr: float | None = None,
    calib_adv_energy: np.ndarray | None = None,
    calib_adv_correct_standard: np.ndarray | None = None,
    calib_adv_correct_robust: np.ndarray | None = None,
    # required for mode == "clean_acc"
    target_clean_acc: float | None = None,
) -> tuple[float, dict]:
    """
    Returns (threshold, info) where info contains empirical FPR and clean
    accuracy figures at the chosen threshold.

    Parameters
    ----------
    mode : "fpr" | "asr" | "clean_acc"
    """
    if mode == "fpr":
        if target_fpr is None:
            raise ValueError("target_fpr required for mode='fpr'")
        threshold = float(np.quantile(clean_energy, 1.0 - target_fpr))

    elif mode == "asr":
        if any(x is None for x in [target_asr, calib_adv_energy,
                                    calib_adv_correct_standard,
                                    calib_adv_correct_robust]):
            raise ValueError(
                "target_asr, calib_adv_energy, calib_adv_correct_standard, "
                "and calib_adv_correct_robust are all required for mode='asr'"
            )
        lo = calib_adv_energy.min() - 1e-6
        hi = calib_adv_energy.max() + 1e-6
        asr_lo = _ensemble_asr(lo, calib_adv_energy,
                               calib_adv_correct_standard, calib_adv_correct_robust)
        asr_hi = _ensemble_asr(hi, calib_adv_energy,
                               calib_adv_correct_standard, calib_adv_correct_robust)
        if not (min(asr_lo, asr_hi) <= target_asr <= max(asr_lo, asr_hi)):
            raise ValueError(
                f"target_asr={target_asr:.4f} outside achievable range "
                f"[{min(asr_lo, asr_hi):.4f}, {max(asr_lo, asr_hi):.4f}]"
            )
        threshold = brentq(
            lambda t: _ensemble_asr(
                t, calib_adv_energy,
                calib_adv_correct_standard, calib_adv_correct_robust,
            ) - target_asr,
            a=lo, b=hi,
        )

    elif mode == "clean_acc":
        if target_clean_acc is None:
            raise ValueError("target_clean_acc required for mode='clean_acc'")
        threshold = brentq(
            lambda t: _ensemble_clean_acc(
                t, clean_energy, clean_correct_robust, base_clean_acc,
            ) - target_clean_acc,
            a=clean_energy.min(),
            b=clean_energy.max(),
        )

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'fpr', 'asr', or 'clean_acc'.")

    # Common post-calibration stats
    clean_flagged          = clean_energy > threshold
    empirical_fpr          = float(clean_flagged.mean())
    retained               = (~clean_flagged) | (clean_flagged & clean_correct_robust)
    clean_acc_ensemble     = float(retained.mean() * base_clean_acc)
    clean_acc_after_reject = float(base_clean_acc * (1.0 - empirical_fpr))

    info = {
        "threshold":            threshold,
        "empirical_fpr":        empirical_fpr,
        "clean_acc_standard":   base_clean_acc,
        "clean_acc_ensemble":   clean_acc_ensemble,
        "clean_acc_reject":     clean_acc_after_reject,
    }
    return threshold, info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_ensemble(
    attack_name: str,
    clean_energy: np.ndarray,
    adv_energy: np.ndarray,
    adv_correct_standard: np.ndarray,
    adv_correct_robust: np.ndarray,
    threshold: float,
) -> dict:
    """
    Routing: energy > threshold → robust model, else → standard model.

    Returns a dict with: attack, asr_raw, asr_post, detection_rate,
    acc_standard, acc_ensemble, auc.
    """
    adv_flagged = adv_energy > threshold
    adv_ensemble_correct = (
        (~adv_flagged & adv_correct_standard) |
        ( adv_flagged & adv_correct_robust)
    )

    labels = np.concatenate([np.zeros(len(clean_energy)), np.ones(len(adv_energy))])
    scores = np.concatenate([clean_energy, adv_energy])

    return {
        "attack":         attack_name,
        "asr_raw":        float((~adv_correct_standard).mean()),
        "asr_post":       float((~adv_ensemble_correct).mean()),
        "detection_rate": float(adv_flagged.mean()),
        "acc_standard":   float(adv_correct_standard.mean()),
        "acc_ensemble":   float(adv_ensemble_correct.mean()),
        "auc":            float(roc_auc_score(labels, scores)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_threshold(
    clean_energy: np.ndarray,
    attack_energies: dict[str, np.ndarray],
    threshold: float,
    empirical_fpr: float,
    layer: str,
    save_path: str | Path,
    title_extra: str = "",
) -> Path:
    """
    KDE density plot with the detection threshold.
    One curve per entry in attack_energies, plus clean.
    """
    palette = ["tomato", "seagreen", "mediumpurple", "darkorange", "deeppink"]
    all_energies = [clean_energy] + list(attack_energies.values())
    x_min = min(e.min() for e in all_energies)
    x_max = max(e.max() for e in all_energies)
    x_grid = np.linspace(x_min, x_max, 500)

    fig, ax = plt.subplots(figsize=(7, 5))

    def _kde(data, label, color):
        if np.std(data) > 1e-8:
            kde = gaussian_kde(data)
            y = kde(x_grid)
            ax.plot(x_grid, y, label=label, color=color)
            ax.fill_between(x_grid, y, alpha=0.15, color=color)

    _kde(clean_energy, "clean", "steelblue")
    for (atk, energy), color in zip(attack_energies.items(), palette):
        _kde(energy, atk, color)

    ax.axvline(
        threshold, color="black", linestyle="--", linewidth=2,
        label=f"threshold = {threshold:.2f}  |  FPR = {empirical_fpr:.2%}",
    )
    ax.axvspan(threshold, x_max, color="gray", alpha=0.12, label="→ robust model")

    title = f"Ensemble detector — {layer}"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title)
    ax.set_xlabel("tail energy")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_calibration_info(info: dict, mode: str, target) -> None:
    print(f"  Calibration mode:                         {mode}  (target = {target})")
    print(f"  Threshold:                                {info['threshold']:.6f}")
    print(f"  Empirical clean FPR:                      {info['empirical_fpr']:.4f}")
    print(f"  Clean accuracy — standard only:           {info['clean_acc_standard']:.4f}")
    print(f"  Clean accuracy — ensemble (route robust): {info['clean_acc_ensemble']:.4f}")
    print(f"  Clean accuracy — detector only (reject):  {info['clean_acc_reject']:.4f}")


def print_results_table(results: list[dict], calib_info: dict) -> None:
    header = (
        f"{'Attack':<12} "
        f"{'AUC':>7} "
        f"{'ASR raw':>9} "
        f"{'Detected':>9} "
        f"{'ASR post':>9} "
        f"{'Acc std':>8} "
        f"{'Acc ens':>8}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['attack']:<12} "
            f"{r['auc']:>7.4f} "
            f"{r['asr_raw']:>9.4f} "
            f"{r['detection_rate']:>9.4f} "
            f"{r['asr_post']:>9.4f} "
            f"{r['acc_standard']:>8.4f} "
            f"{r['acc_ensemble']:>8.4f}"
        )
    print(sep)
    print()
    print(f"  Clean FPR:                    {calib_info['empirical_fpr']:.2%}")
    print(f"  Clean accuracy — standard:    {calib_info['clean_acc_standard']:.2%}")
    print(f"  Clean accuracy — ensemble:    {calib_info['clean_acc_ensemble']:.2%}")