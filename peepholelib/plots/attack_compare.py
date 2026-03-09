from pathlib import Path
import torch
import matplotlib.pyplot as plt
from tensordict import PersistentTensorDict


class ThreatModelComparator:
    """
    Compare attacks within each threat model and save plots.

    Expected files:
        Clean vs Attack obv

    Example:
      threat_models = {
          "Linf": ["BIM", "APGD", "PGD"],
          "L2":   ["CW", "DeepFool"],
      }
    """

    def __init__(self, base_path, dataset="CIFAR100", threat_models=None):
        self.base = Path(base_path)
        self.dataset = dataset
        self.threat_models = threat_models or {}

    def _load(self, name):
        return PersistentTensorDict.from_h5(self.base / name, mode="r")

    def _clean(self, split):
        return self._load(f"dss.{self.dataset}-{split}")

    def _adv(self, split, attack):
        return self._load(f"dss.{attack}-{self.dataset}-{split}")

    # -----------------------
    # Metrics
    # -----------------------
    def _correct_mask(self, td):
        if "result" in td.keys():
            return td["result"][:].bool()
        if ("pred" in td.keys()) and ("label" in td.keys()):
            return (td["pred"][:] == td["label"][:])
        return None

    def _attack_success_mask(self, td):
        if "attack_success" in td.keys():
            return (td["attack_success"][:] > 0.5)
        return None

    def _robust_acc(self, td):
        m = self._correct_mask(td)
        if m is None:
            return None
        return float(m.float().mean().item())

    # def _conditional_asr(self, clean, adv):
    #     """
    #     ASR
    #     """
    #     clean_corr = self._correct_mask(clean)
    #     if clean_corr is None:
    #         return None

    #     adv_corr = self._correct_mask(adv)
    #     if adv_corr is not None:
    #         return float((~adv_corr[clean_corr]).float().mean().item())

    #     adv_succ = self._attack_success_mask(adv)
    #     if adv_succ is not None:
    #         return float(adv_succ[clean_corr].float().mean().item())

    #     return None
    def _conditional_misclf(self, clean, adv):
        """
        Misclassification rate | clean-correct (untargeted)
        """
        clean_corr = self._correct_mask(clean)
        adv_corr = self._correct_mask(adv)
        if clean_corr is None or adv_corr is None:
            return None
        return float((~adv_corr[clean_corr]).float().mean().item())

    def _conditional_targeted_asr(self, clean, adv):
        """
        Targeted success rate | clean-correct
        """
        clean_corr = self._correct_mask(clean)
        if clean_corr is None:
            return None

        adv_succ = self._attack_success_mask(adv)
        if adv_succ is None:
            return None

        return float(adv_succ[clean_corr].float().mean().item())

    def _norms(self, clean, adv):
        x0 = clean["image"][:].float()
        x1 = adv["image"][:].float()
        d = (x1 - x0).view(x0.shape[0], -1)
        linf = d.abs().max(dim=1).values
        l2 = torch.sqrt((d * d).sum(dim=1))
        return linf.detach().cpu(), l2.detach().cpu()

    def _summary(self, x):
        x = x.float()
        return {
            "mean": float(x.mean().item()),
            "median": float(x.median().item()),
            "p95": float(x.quantile(0.95).item()),
            "max": float(x.max().item()),
        }

    def evaluate(self, split):
        clean = self._clean(split)
        results = {}

        for tm_name, attacks in self.threat_models.items():
            results[tm_name] = {}
            for atk in attacks:
                adv = self._adv(split, atk)
                linf, l2 = self._norms(clean, adv)

                results[tm_name][atk] = {
                    "robust_acc": self._robust_acc(adv),
                    "targeted_asr_clean_correct": self._conditional_targeted_asr(clean, adv),
                    "misclf_clean_correct": self._conditional_misclf(clean, adv),
                    "linf": self._summary(linf),
                    "l2": self._summary(l2),
                    # store vectors for plotting distributions
                    "_linf_vec": linf,
                    "_l2_vec": l2,
                }

        return results

    # -----------------------
    # Plotting
    # -----------------------
    def plot(self, split, out_dir):
        """
        Saves one main figure per threat model:
          - Linf: bars of robust_acc/ASR + boxplot of ||δ||_inf
          - L2:   bars of robust_acc/ASR + boxplot of ||δ||_2
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = self.evaluate(split)

        for tm_name, atk_dict in results.items():
            attacks = list(atk_dict.keys())

            # gather scalar metrics (handle None)
            racc = [atk_dict[a]["robust_acc"] for a in attacks]
            # asr  = [atk_dict[a]["asr_clean_correct"] for a in attacks]
            asr = [atk_dict[a]["targeted_asr_clean_correct"] for a in attacks]

            # choose primary norm by threat model name
            is_linf = tm_name.lower() in ["linf", "l_infty", "l∞", "l-infty", "l-infinity", "linfinity"]
            primary_vecs = [atk_dict[a]["_linf_vec"] if is_linf else atk_dict[a]["_l2_vec"] for a in attacks]
            primary_label = "$\\|\\delta\\|_\\infty$" if is_linf else "$\\|\\delta\\|_2$"
            
            # --- Figure layout ---
            fig = plt.figure(figsize=(12, 4.5))
            gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])

            ax_bar = fig.add_subplot(gs[0, 0])
            ax_box = fig.add_subplot(gs[0, 1])

            x = list(range(len(attacks)))
            width = 0.35

            # barplot: robust acc + ASR if available
            # If None, we skip that series.
            any_racc = any(v is not None for v in racc)
            any_asr  = any(v is not None for v in asr)

            if any_racc:
                ax_bar.bar([i - width/2 for i in x],
                           [v if v is not None else 0.0 for v in racc],
                           width=width, label="Robust accuracy")

            if any_asr:
                ax_bar.bar([i + width/2 for i in x],
                           [v if v is not None else 0.0 for v in asr],
                           width=width, label=r"ASR")

            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(attacks, rotation=0)
            ax_bar.set_ylim(0.0, 1.0)
            ax_bar.set_ylabel("Rate")
            ax_bar.set_title(f"{tm_name} — metrics ({split})")
            ax_bar.grid(True, axis="y", linestyle=":")
            if any_racc or any_asr:
                ax_bar.legend()

            # boxplot
            ax_box.boxplot(primary_vecs, labels=attacks, showfliers=False)
            ax_box.set_title(f"{tm_name} — {primary_label} distribution ({split})")
            ax_box.set_ylabel(primary_label)
            ax_box.grid(True, axis="y", linestyle=":")

            fig.tight_layout()
            fig.savefig(out_dir / f"compare_{tm_name}_{split}.png", dpi=200)
            plt.close(fig)

        # clean up
        for tm in results:
            for a in results[tm]:
                results[tm][a].pop("_linf_vec", None)
                results[tm][a].pop("_l2_vec", None)

        return results