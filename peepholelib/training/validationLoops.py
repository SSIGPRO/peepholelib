from time import time
from math import isinf
import torch

def DefaultValidationLoop(**kwargs):
    trainer = kwargs['trainer']
    epoch = kwargs['epoch']
    t0 = kwargs.get('t0', None)
    # validation
    trainer.model._model.eval()
    with torch.no_grad():
        loss_acc = 0.0
        acc_acc = 0.0
        samples_acc = 0

        for _, _data in zip(range(trainer.iter_val), trainer.val_dl):
            data = trainer.in_parser(_data)
            images = data["image"].contiguous().to(trainer.device, non_blocking=True)
            labels = data["label"].contiguous().to(trainer.device, non_blocking=True)
            n_samples = len(images)
            samples_acc += n_samples

            model_out = trainer.model(images)
            pred = trainer.out_parser(model_out)
            loss = trainer.loss_fn(pred, labels)

            loss_acc += loss * n_samples
            acc_acc += trainer.acc_fn(pred, labels)

        trainer.val_losses[epoch] = (loss_acc / samples_acc).detach().cpu()
        trainer.val_acc[epoch] = (acc_acc / samples_acc).detach().cpu()

    # step the scheduler
    if trainer.scheduler is not None:
        if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            old_lr = trainer.optim.param_groups[0]["lr"]
            trainer.scheduler.step(trainer.val_losses[epoch])
            current_lr = trainer.optim.param_groups[0]["lr"]

            if old_lr != current_lr and trainer.verbose:
                print(f'New LR: {current_lr:.6f}')
        else:
            trainer.scheduler.step()

    # best model tracking
    if trainer.val_losses[epoch] < trainer.best_val_loss:
        trainer.best_val_loss = trainer.val_losses[epoch]
        trainer.best_epoch = epoch
        if trainer.num_bad_epochs is not None : trainer.num_bad_epochs = 0

        # Optional: save best model snapshot
        best_model_path = getattr(trainer, "best_model_path", trainer.path / "best_model")
        best_model_path.mkdir(parents=True, exist_ok=True)
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
            "num_bad_epochs": trainer.num_bad_epochs,
        }
        torch.save(_d, best_model_path / "best_model_config.pt")

        if trainer.verbose:
            print(f'  -> New best validation loss: {trainer.best_val_loss:.6f}')
    else:
        if trainer.num_bad_epochs is not None : trainer.num_bad_epochs += 1

    # early stopping
    if trainer.early_stopping:
        if (hasattr(trainer.scheduler, "num_bad_epochs") and hasattr(trainer.scheduler, "patience")):
            if trainer.scheduler.num_bad_epochs >= trainer.scheduler.patience:
                if trainer.verbose:
                    print(
                        f'Early stopping: no improvement for {trainer.scheduler.num_bad_epochs} epochs '
                        f'(patience={trainer.scheduler.patience}).'
                    )
                return True
        else:
            if trainer.num_bad_epochs >= trainer.early_stopping_patience:
                if trainer.verbose:
                    print(
                        f'Early stopping: no improvement for {trainer.num_bad_epochs} epochs '
                        f'(patience={trainer.early_stopping_patience}).'
                    )
                return True

    if trainer.verbose and t0 is not None:
        print(
            f'epoch {epoch} - train loss: {trainer.train_losses[epoch]:.4f} - '
            f'val loss: {trainer.val_losses[epoch]:.4f} - '
            f'train acc: {trainer.train_acc[epoch]*100:.2f} - '
            f'val acc: {trainer.val_acc[epoch]*100:.2f} - '
            f'time: {time()-t0:.2f}'
        )

    return False