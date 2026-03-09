from pathlib import Path
import matplotlib.pyplot as plt
from tensordict import PersistentTensorDict


class AttackDatasetEvaluator:

    def __init__(self, base_path, dataset_name="CIFAR100", split="val"):
        self.base_path = Path(base_path)
        self.dataset_name = dataset_name
        self.split = split
        self.results = {}

    def _load_dataset(self, name):
        if name == self.dataset_name:
            path = self.base_path / f"dss.{name}-{self.split}"
        else:
            path = self.base_path / f"dss.{name}-{self.dataset_name}-{self.split}"
        return PersistentTensorDict.from_h5(path, mode="r")

    def _accuracy(self, td):
        y = td["label"][:]
        p = td["pred"][:]
        return (p == y).float().mean().item()

    def evaluate(self, attack_names):
        self.results = {}

        for name in attack_names:
            print(f"Evaluating {name}")
            td = self._load_dataset(name)
            self.results[name] = self._accuracy(td)

        return self.results

    def plot(self, save_dir="plots", filename="attack_accuracy.png"):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        names = list(self.results.keys())
        values = list(self.results.values())

        plt.figure()
        plt.bar(names, values)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title(f"{self.split.capitalize()} Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_dir / filename, dpi=200, bbox_inches="tight")
        plt.close()

        print("Saved plot to", save_dir / filename)