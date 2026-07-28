def default_train_loop(self, **kwargs):
    """
    Default training loop: runs one training epoch using the pre-built
    `DataLoader` for the train split.

    Args:
    - epoch (int): current epoch index, passed by `Trainer._train_epoch`.
    - dataset_key (str): key into `self._dls`/`self._iters` for the train split. Defaults to `'train'`.
    """
    epoch = kwargs['epoch']
    dataset_key = kwargs.get('dataset_key', 'train')

    train_dl = self._dls[dataset_key]
    iter_train = self._iters[dataset_key]

    loss_acc = 0.0
    acc_acc = 0.0
    samples_acc = 0
    self.model._model.train()

    for it, _data in zip(range(iter_train), train_dl):
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

    self.optim.zero_grad()
    loss_mean.backward()
    self.optim.step()

    self.train_losses[epoch] = (loss_mean).detach().cpu()
    self.train_acc[epoch] = (acc_mean).detach().cpu()
    return
