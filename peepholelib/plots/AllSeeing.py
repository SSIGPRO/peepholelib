from pathlib import Path
import matplotlib.pyplot as plt
from tensordict import PersistentTensorDict


class AttackExampleVisualizer:
    """
    Visualize clean images, adversarial images, and perturbations
    for exact indices chosen by the user.
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
    # Image helpers
    # -----------------------
    def _to_display_image(self, x):
        return x.detach().cpu().float().permute(1, 2, 0).numpy()

    def _to_display_delta(self, delta, gain=8.0):
        delta = delta.detach().cpu().float().permute(1, 2, 0)
        delta = delta * gain
        delta = delta + 0.5
        return delta.clamp(0.0, 1.0).numpy()

    # -----------------------
    # Utility
    # -----------------------
    def _normalize_indices(self, indices):
        if isinstance(indices, int):
            return [indices]
        if isinstance(indices, (list, tuple)):
            return list(indices)
        raise TypeError("indices must be an int, list, or tuple")

    # -----------------------
    # Visualization
    # -----------------------
    def save_examples(
        self,
        split,
        attack,
        out_dir,
        indices,
    ):
        """
        Save clean image, adversarial image, and perturbation
        for the exact index or indices provided.

        Args:
            split (str): dataset split, e.g. 'test'
            attack (str): attack name, e.g. 'APGDf'
            out_dir (str | Path): output directory
            indices (int | list[int]): exact sample index or indices
        """
        clean = self._clean(split)
        adv = self._adv(split, attack)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        indices = self._normalize_indices(indices)

        clean_images = clean["image"][:]
        adv_images = adv["image"][:]

        n_total = clean_images.shape[0]
        saved_paths = []

        for i in indices:
            if not (0 <= i < n_total):
                raise IndexError(f"Index {i} is out of bounds for dataset of size {n_total}")

            x = clean_images[i]
            x_adv = adv_images[i]
            delta = x_adv - x

            fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

            # Clean image
            axes[0].imshow(self._to_display_image(x))
            axes[0].axis("off")

            # Adversarial image
            axes[1].imshow(self._to_display_image(x_adv))
            axes[1].axis("off")

            # Perturbation
            axes[2].imshow(self._to_display_delta(delta, gain=8.0))
            axes[2].axis("off")

            fig.tight_layout()
            save_path = out_dir / f"{attack}_{split}_idx{i}.png"
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            saved_paths.append(save_path)

        return saved_paths