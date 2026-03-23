from pathlib import Path
import torch
import matplotlib.pyplot as plt
from tensordict import PersistentTensorDict


class AttackExampleVisualizer:
    """
    Visualize clean images, adversarial images, and perturbations
    for a given attack dataset.
    """

    def __init__(
        self,
        base_path,
        dataset="CIFAR100",
        class_names=None,
    ):
        self.base = Path(base_path)
        self.dataset = dataset
        self.class_names = class_names

    # -----------------------
    # Loading
    # -----------------------
    def _load(self, name):
        return PersistentTensorDict.from_h5(self.base / name, mode="r")

    def _clean(self, split):
        return self._load(f"dss.{self.dataset}-{split}")

    def _adv(self, split, attack):
        return self._load(f"dss.{attack}-{self.dataset}-{split}")

    # -----------------------
    # Metadata helpers
    # -----------------------
    def _correct_mask(self, td):
        if "result" in td.keys():
            return td["result"][:].bool()
        if ("pred" in td.keys()) and ("label" in td.keys()):
            return td["pred"][:] == td["label"][:]
        return None

    def _attack_success_mask(self, td):
        if "attack_success" in td.keys():
            return td["attack_success"][:] > 0.5
        return None

    def _get_labels(self, td):
        return td["label"][:] if "label" in td.keys() else None

    def _get_preds(self, td):
        return td["pred"][:] if "pred" in td.keys() else None

    def _label_to_name(self, y):
        y = int(y)
        if self.class_names is None:
            return str(y)
        return self.class_names[y]

    # -----------------------
    # Image helpers
    # -----------------------
    def _to_display_image(self, x):
        return x.detach().cpu().float().permute(1, 2, 0).numpy()

    # def _to_display_delta(self, delta):
    #     return delta.detach().cpu().float().permute(1, 2, 0).numpy()
    # Commented as we see nothing
    def _to_display_delta(self, delta, gain=8.0):
        delta = delta.detach().cpu().float()
        delta = delta.permute(1, 2, 0)

        delta = delta * gain          # enlarge perturbation
        delta = delta + 0.5           # zero perturbation -> gray

        return delta.clamp(0.0, 1.0).numpy()
    # -----------------------
    # Selection
    # -----------------------
    def select_indices(self, split, attack, n=5, mode="successful"):
        """
        mode:
          - 'successful': clean-correct and attack successful
          - 'misclassified': clean-correct and adv-wrong
          - 'changed': clean prediction != adv prediction
          - 'wrong_clean': already wrong on clean image
          - 'all': first n samples
        """
        clean = self._clean(split)
        adv = self._adv(split, attack)

        n_total = clean["image"].shape[0]
        idx = torch.arange(n_total)

        clean_corr = self._correct_mask(clean)
        adv_corr = self._correct_mask(adv)
        adv_succ = self._attack_success_mask(adv)
        clean_pred = self._get_preds(clean)
        adv_pred = self._get_preds(adv)

        if mode == "successful":
            if clean_corr is not None and adv_succ is not None:
                mask = clean_corr & adv_succ
            elif clean_corr is not None and adv_corr is not None:
                mask = clean_corr & (~adv_corr)
            else:
                mask = torch.ones(n_total, dtype=torch.bool)

        elif mode == "misclassified":
            if clean_corr is not None and adv_corr is not None:
                mask = clean_corr & (~adv_corr)
            else:
                mask = torch.ones(n_total, dtype=torch.bool)

        elif mode == "changed":
            if clean_pred is not None and adv_pred is not None:
                mask = clean_pred != adv_pred
            else:
                mask = torch.ones(n_total, dtype=torch.bool)

        elif mode == "wrong_clean":
            if clean_corr is not None:
                mask = ~clean_corr
            else:
                mask = torch.ones(n_total, dtype=torch.bool)

        else:
            mask = torch.ones(n_total, dtype=torch.bool)

        return idx[mask][:n].tolist()

    # -----------------------
    # Visualization
    # -----------------------
    def save_examples(
        self,
        split,
        attack,
        out_dir,
        indices=None,
        n=5,
        mode="successful",
        show_titles=True,
    ):
        clean = self._clean(split)
        adv = self._adv(split, attack)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if indices is None:
            indices = self.select_indices(split=split, attack=attack, n=n, mode=mode)

        clean_images = clean["image"][:]
        adv_images = adv["image"][:]
        labels = self._get_labels(clean)
        clean_preds = self._get_preds(clean)
        adv_preds = self._get_preds(adv)
        adv_succ = self._attack_success_mask(adv)

        saved_paths = []

        for i in indices:
            x = clean_images[i]
            x_adv = adv_images[i]
            delta = x_adv - x

            fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

            axes[0].imshow(self._to_display_image(x))
            axes[0].axis("off")
            if show_titles:
                title = "Clean"
                if labels is not None:
                    title += f"\nlabel: {self._label_to_name(labels[i])}"
                if clean_preds is not None:
                    title += f"\npred: {self._label_to_name(clean_preds[i])}"
                axes[0].set_title(title)

            axes[1].imshow(self._to_display_image(x_adv))
            axes[1].axis("off")
            if show_titles:
                title = f"Adversarial ({attack})"
                if adv_preds is not None:
                    title += f"\npred: {self._label_to_name(adv_preds[i])}"
                if adv_succ is not None:
                    title += f"\nsuccess: {bool(adv_succ[i].item())}"
                axes[1].set_title(title)

            axes[2].imshow(self._to_display_delta(delta, gain=8.0))
            axes[2].axis("off")
            if show_titles:
                linf = delta.abs().max().item()
                axes[2].set_title(f"Perturbation (x8)\nL∞={linf:.4f}")

            fig.tight_layout()
            save_path = out_dir / f"{attack}_{split}_idx{i}.png"
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            saved_paths.append(save_path)

        return saved_paths