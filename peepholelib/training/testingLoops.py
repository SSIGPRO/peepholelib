# peepholelib/training/testingLoops.py
import torch

def DefaultTestLoop(**kwargs):
    trainer = kwargs['trainer']
    trainer.model._model.eval()
    loss_acc = 0.0
    acc_acc = 0.0

    n_samples = len(trainer.test_dl.dataset)
    with torch.no_grad():
        for _data in trainer.test_dl:
            data = trainer.in_parser(_data)
            images = data["image"].contiguous().to(trainer.device, non_blocking=True)
            labels = data["label"].contiguous().to(trainer.device, non_blocking=True)

            model_out = trainer.model(images)
            pred = trainer.out_parser(model_out)

            loss_acc += trainer.loss_fn(pred, labels)
            acc_acc += trainer.acc_fn(pred, labels)

    test_loss = (loss_acc/n_samples).detach().cpu()
    test_acc = (acc_acc/n_samples).detach().cpu()

    if trainer.verbose:
        print(f"test loss: {test_loss:.4f} - test acc: {test_acc*100:.2f}")

    return test_loss, test_acc
