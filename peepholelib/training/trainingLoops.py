def DefaultTrainLoop(**kwargs):
    trainer = kwargs['trainer']
    epoch = kwargs['epoch']
    loss_acc = 0.0
    acc_acc = 0.0
    samples_acc = 0
    trainer.model._model.train()

    for it, _data in zip(range(trainer.iter_train), trainer.train_dl):
        data = trainer.in_parser(_data)
        images = data["image"].contiguous().to(trainer.device, non_blocking=True)
        labels = data["label"].contiguous().to(trainer.device, non_blocking=True)
        n_samples = len(images)
        samples_acc += n_samples

        model_out = trainer.model(images)
        pred = trainer.out_parser(model_out)

        loss_acc += trainer.loss_fn(pred, labels)
        acc_acc += trainer.acc_fn(pred, labels)

    loss_mean = loss_acc/samples_acc
    acc_mean = acc_acc/samples_acc

    trainer.optim.zero_grad()
    loss_mean.backward()
    trainer.optim.step()

    trainer.train_losses[epoch] = (loss_mean).detach().cpu()
    trainer.train_acc[epoch] = (acc_mean).detach().cpu()
    return
