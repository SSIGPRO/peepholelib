from time import time
import torch

def default_val_loop(self, **kwargs):
    epoch = kwargs['epoch']

    self.model._model.eval()
    with torch.no_grad():
        loss_acc = 0.0
        acc_acc = 0.0
        samples_acc = 0

        for _, _data in zip(range(self.iter_val), self.val_dl):
            data = self.in_parser(_data)
            images = data["image"].contiguous().to(self.device, non_blocking=True)
            labels = data["label"].contiguous().to(self.device, non_blocking=True)
            n_samples = len(images)
            samples_acc += n_samples

            model_out = self.model(images)
            pred = self.out_parser(model_out)

            loss_acc += self.loss_fn(pred, labels)
            acc_acc += self.acc_fn(pred, labels)

        loss_mean = loss_acc/samples_acc
        acc_mean = acc_acc/samples_acc

        self.val_losses[epoch] = (loss_mean).detach().cpu()
        self.val_acc[epoch] = (acc_mean).detach().cpu()

    # step the scheduler
    if self.scheduler is not None:
        old_lr = self.scheduler.get_last_lr()

        #TODO: this is a bit ugly that each step might have a different interface
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(self.val_losses[epoch])
        else:
            self.scheduler.step()

        current_lr = self.scheduler.get_last_lr()

        if old_lr != current_lr and self.verbose:
            print(f'New LR: {current_lr[-1]:.6f}')

    # best model tracking
    if self.val_losses[epoch] < self.best_val_loss:
        self.best_val_loss = self.val_losses[epoch]
        self.best_epoch = epoch
        self.num_bad_epochs = 0

        # Optional: save best model snapshot
        self.save_fn(
                epoch = epoch,
                file = self.best_model_file,
                plot = False
                )

        if self.verbose:
            print(f'  -> New best validation loss: {self.best_val_loss:.6f}')
    else:
        self.num_bad_epochs += 1

    # early stopping
    if self.num_bad_epochs > self.early_stopping_patience:
        if self.verbose:
            print(
                f'Early stopping: no improvement for {self.num_bad_epochs} epochs '
                f'(patience={self.early_stopping_patience}).'
            )
        return True

    return False
