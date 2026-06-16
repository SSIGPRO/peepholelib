# general python
from pathlib import Path

# torch stuff
import torch

# plotting stuff
from matplotlib import pyplot as plt

def default_save(self, **kwargs):
    epoch = kwargs['epoch']
    file = kwargs['file']
    plot = kwargs.get('plot', False)

    _d = {
        "epoch": epoch,
        "train_losses": self.train_losses[: epoch + 1],
        "train_accuracy": self.train_acc[: epoch + 1],
        "val_losses": self.val_losses[: epoch + 1],
        "val_accuracy": self.val_acc[: epoch + 1],
        "state_dict": self.model._model.state_dict(),
        "optimizer": self.optim.state_dict(),
        "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        "best_epoch": self.best_epoch,
        "best_val_loss": self.best_val_loss,
    }
    torch.save(_d, file)

    # skipping plotting
    if not plot:
        return

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    train_losses_np = self.train_losses[: epoch + 1].detach().cpu().numpy()
    val_losses_np = self.val_losses[: epoch + 1].detach().cpu().numpy()
    train_acc_np = self.train_acc[: epoch + 1].detach().cpu().numpy()
    val_acc_np = self.val_acc[: epoch + 1].detach().cpu().numpy()

    axs[0].plot(train_losses_np, label="loss_train")
    axs[0].plot(val_losses_np, label="loss_val")
    axs[0].set_ylabel("loss")
    axs[0].set_title("Loss")

    axs[1].plot(train_acc_np * 100, label="train")
    axs[1].plot(val_acc_np * 100, label="val")
    axs[1].set_ylabel("Acc")
    axs[1].set_xlabel("epoch")
    axs[1].set_title("Accuracy")

    # Highlight best model epoch with a star on each curve.
    axs[0].plot(
        [self.best_epoch],
        [train_losses_np[self.best_epoch]],
        marker="*",
        markersize=12,
        linestyle="None",
        color=axs[0].lines[0].get_color(),
    )
    axs[0].plot(
        [self.best_epoch],
        [val_losses_np[self.best_epoch]],
        marker="*",
        markersize=12,
        linestyle="None",
        color=axs[0].lines[1].get_color(),
        label=f"best loss {val_losses_np[self.best_epoch]:.3f}",
    )
    axs[1].plot(
        [self.best_epoch],
        [train_acc_np[self.best_epoch] * 100],
        marker="*",
        markersize=12,
        linestyle="None",
        color=axs[1].lines[0].get_color(),
    )
    axs[1].plot(
        [self.best_epoch],
        [val_acc_np[self.best_epoch] * 100],
        marker="*",
        markersize=12,
        linestyle="None",
        color=axs[1].lines[1].get_color(),
        label=f"best Acc {val_acc_np[self.best_epoch]*100:.2f}",
    )

    for ax in axs:
        ax.semilogy()
        ax.legend()

    fig.savefig(self.loss_plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return
