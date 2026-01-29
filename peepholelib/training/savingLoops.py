import torch
from matplotlib import pyplot as plt

def DefaultSavingLoop(trainer, epoch):
    _d = {
        "epoch": epoch,
        "train_losses": trainer.train_losses[: epoch + 1],
        "train_accuracy": trainer.train_acc[: epoch + 1],
        "val_losses": trainer.val_losses[: epoch + 1],
        "val_accuracy": trainer.val_acc[: epoch + 1],
        "state_dict": trainer.model._model.state_dict(),
        "optimizer": trainer.optim.state_dict(),
        "scheduler": trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
        "best_epoch": trainer.best_epoch,
        "best_val_loss": trainer.best_val_loss,
    }
    torch.save(_d, trainer.file.as_posix() + "." + str(epoch) + ".pt")

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    train_losses_np = trainer.train_losses[: epoch + 1].detach().cpu().numpy()
    val_losses_np = trainer.val_losses[: epoch + 1].detach().cpu().numpy()
    train_acc_np = trainer.train_acc[: epoch + 1].detach().cpu().numpy()
    val_acc_np = trainer.val_acc[: epoch + 1].detach().cpu().numpy()

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
        [trainer.best_epoch],
        [train_losses_np[trainer.best_epoch]],
        marker="*",
        markersize=12,
        linestyle="None",
        color=axs[0].lines[0].get_color(),
    )
    axs[0].plot(
        [trainer.best_epoch],
        [val_losses_np[trainer.best_epoch]],
        marker="*",
        markersize=12,
        linestyle="None",
        color=axs[0].lines[1].get_color(),
        label=f"best loss {val_losses_np[trainer.best_epoch]:.3f}",
    )
    axs[1].plot(
        [trainer.best_epoch],
        [train_acc_np[trainer.best_epoch] * 100],
        marker="*",
        markersize=12,
        linestyle="None",
        color=axs[1].lines[0].get_color(),
    )
    axs[1].plot(
        [trainer.best_epoch],
        [val_acc_np[trainer.best_epoch] * 100],
        marker="*",
        markersize=12,
        linestyle="None",
        color=axs[1].lines[1].get_color(),
        label=f"best Acc {val_acc_np[trainer.best_epoch]*100:.2f}",
    )

    for ax in axs:
        ax.semilogy()
        ax.legend()

    plot_path = trainer.file.as_posix() + ".losses.png"
    if not getattr(trainer, "_plot_archived", False):
        try:
            from pathlib import Path

            plot_file = Path(plot_path)
            if plot_file.exists():
                archived = plot_file.with_name(
                    plot_file.stem + f".prev.phase_{trainer.phase_num}" + plot_file.suffix
                )
                plot_file.rename(archived)
        finally:
            trainer._plot_archived = True

    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
