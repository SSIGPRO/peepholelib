# peepholelib/training/testingLoops.py
import torch

def DefaultTestLoop(**kwargs):
    trainer = kwargs['trainer']
    trainer.model._model.eval()
    loss_acc = 0.0
    acc_acc = 0.0
    samples_acc = 0

    with torch.no_grad():
        for _data in trainer.test_dl:
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

    test_loss = (loss_acc / samples_acc).detach().cpu()
    test_acc = (acc_acc / samples_acc).detach().cpu()

    if trainer.verbose:
        print(f"test loss: {test_loss:.4f} - test acc: {test_acc*100:.2f}")

    return test_loss, test_acc
