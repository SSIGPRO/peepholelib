# peepholelib/training/testingLoops.py
import torch

def default_test_loop(self, **kwargs):
    """
    Default test loop: evaluates the model on the pre-built `DataLoader`
    for the test split.

    Args:
    - dataset_key (str): key into `self._dls` for the test split. Defaults to `'test'`.
    """
    dataset_key = kwargs.get('dataset_key', 'test')

    test_dl = self._dls[dataset_key]

    self.model._model.eval()
    loss_acc = 0.0
    acc_acc = 0.0

    n_samples = len(test_dl.dataset)
    with torch.no_grad():
        for _data in test_dl:
            data = self.in_parser(_data)
            images = data["image"].contiguous().to(self.device, non_blocking=True)
            labels = data["label"].contiguous().to(self.device, non_blocking=True)

            model_out = self.model(images)
            pred = self.out_parser(model_out)

            loss_acc += self.loss_fn(pred, labels)
            acc_acc += self.acc_fn(pred, labels)

    test_loss = (loss_acc/n_samples).detach().cpu()
    test_acc = (acc_acc/n_samples).detach().cpu()

    if self.verbose:
        print(f"test loss: {test_loss:.4f} - test acc: {test_acc*100:.2f}")

    return test_loss, test_acc
