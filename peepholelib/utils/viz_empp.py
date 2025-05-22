import os
from matplotlib import pyplot as plt
import numpy as np

def empirical_posterior_heatmaps(drillers, dir):
    os.makedirs(dir, exist_ok=True)

    for module_name, driller in drillers.items():
        driller.load()

        if not hasattr(driller, "_empp") or driller._empp is None:
            print(f"No empirical posterior found for {module_name}, skipping.")
            continue

        empp = driller._empp  # shape: [n_clusters, n_classes]
        print(f"Empirical Posterior shape for {module_name}: {empp.shape}")

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(empp.cpu().numpy(), aspect='auto', cmap='viridis', interpolation='nearest')

        ax.set_xlabel("Model Classes")
        ax.set_ylabel("Cluster Index")
        ax.set_title(f"Empirical Posterior - {module_name}")

        ax.set_xticks(np.linspace(0, empp.shape[1] - 1, min(20, empp.shape[1]), dtype=int))
        ax.set_yticks(np.linspace(0, empp.shape[0] - 1, min(20, empp.shape[0]), dtype=int))

        fig.colorbar(im, ax=ax, label="P(g|c)")

        filename = f"empirical_posterior_{module_name.replace('.', '_')}.png"
        filepath = os.path.join(dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved: {filepath}")