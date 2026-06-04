def default_train_loop(self, **kwargs):
    epoch = kwargs['epoch']
    loss_acc = 0.0
    acc_acc = 0.0
    samples_acc = 0
    self.model._model.train()

    for it, _data in zip(range(self.iter_train), self.train_dl):
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
