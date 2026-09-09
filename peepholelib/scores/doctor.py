from math import ceil
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax as sm
from peepholelib.scores.score import Score


class DOCTORScore(Score):
    '''
    Compute the DOCTOR score described in https://arxiv.org/pdf/2106.02395

    When `magnitude` is 0 the scores are computed directly from the parsed outputs, otherwise the inputs are perturbed and passed through the model again.

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed datasets.
    - model (peepholelib.models.model_wrap.ModelWrap): wrapped model, used to compute the logits of the perturbed inputs. Only required when `magnitude != 0`.
    - loaders (list[str]): loaders to consider. If `None`, gets all loaders in `datasets._dss`. Defaults to `None`.
    - temperature (float): temperature factor. Defaults to 1.0.
    - magnitude (float): magnitude of the adversarial perturbation. Defaults to 0.0.
    - batch_size (int): batch size used to compute the scores. Defaults to 128.
    - n_threads (int): `num_workers` passed to `torch.utils.data.DataLoader`. Defaults to 32.
    - input_key (str): key used to read the input images from the dataset. Defaults to `'image'`.
    - output_key (str): key used to read the model outputs from the dataset. Defaults to `'output'`.
    - verbose (bool): print progress messages.
    '''

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'DOCTOR')
        Score.__init__(self, **kwargs)
        return

    def compute(self, **kwargs):
        dss = kwargs['datasets']
        model = kwargs.get('model', None)
        loaders = kwargs.get('loaders') or list(dss._dss.keys())
        temperature = kwargs.get('temperature', 1.0)
        magnitude = kwargs.get('magnitude', 0.0)
        bs = kwargs.get('batch_size', 128)
        n_threads = kwargs.get('n_threads', 32)
        input_key = kwargs.get('input_key', 'image')
        output_key = kwargs.get('output_key', 'output')
        verbose = kwargs.get('verbose', False)

        # skip the loaders already computed
        loaders = [k for k in loaders if not self._is_computed(ds_key=k)]

        # the model is only used with the perturbed inputs
        if magnitude != 0:
            device = model.device

        for ds_key in loaders:
            if verbose: print('Computing', self.name, 'for dataset', ds_key)

            _dss = dss._dss[ds_key]
            n_samples = len(_dss)

            if magnitude == 0:
                logits = _dss[output_key]
            else:
                n_classes = _dss[0:1][output_key].shape[-1]
                logits = torch.empty(n_samples, n_classes)

                dl = DataLoader(
                        _dss,
                        batch_size = bs,
                        shuffle = False,
                        collate_fn = lambda x: x,
                        num_workers = n_threads
                        )

                write_ptr = 0
                for data in tqdm(dl, disable=not verbose, total=ceil(n_samples/bs), desc=f'{self.name} [{ds_key}]'):
                    inputs = data[input_key].to(device)
                    inputs.requires_grad_(True)
                    model._model.zero_grad()

                    output = model(inputs)
                    _scores = sm(output/temperature, dim=1).pow(2).sum(dim=1)
                    _scores.clamp(min=1e-12).log().sum().backward()

                    new_inputs = (inputs + magnitude*inputs.grad.sign()).clamp(0, 1).detach()

                    inputs.requires_grad_(False)
                    model._model.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        output = model(new_inputs)

                    bsz = output.shape[0]
                    logits[write_ptr:write_ptr+bsz] = output.detach().cpu()
                    write_ptr += bsz

            scores = sm(logits/temperature, dim=1).pow(2).sum(dim=1).detach().cpu().reshape(-1)
            self._record(ds_key=ds_key, scores=scores)

        return self._df

    def _save_fitting(self, **kwargs):
        return

    def load(self, **kwargs):
        return 1
