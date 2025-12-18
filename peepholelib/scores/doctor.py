from math import ceil
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax as sm
from peepholelib.scores.score import Score


class DOCTORScore(Score):
    score_name = 'DOCTOR'

    def _compute(self, **kwargs):
        dss = kwargs['datasets']
        model = kwargs['model']
        loaders = kwargs.get('loaders') or list(dss._dss.keys())
        temperature = kwargs.get('temperature', 1.0)
        magnitude = kwargs.get('magnitude', 0.0)
        n_threads = kwargs.get('n_threads', 32)
        bs = kwargs.get('batch_size', 128)
        score_name = kwargs.get('score_name', self.score_name)
        verbose = kwargs.get('verbose', False)

        device = model.device

        for ds_key in loaders:
            if self._is_computed(ds_key, score_name):
                if verbose:
                    print(ds_key, score_name, 'already computed, skipping')
                continue
            if verbose:
                print('Computing', score_name, 'for dataset', ds_key)

            dssds = dss._dss[ds_key]
            n_samples = len(dssds)
            ds_scores = torch.empty(n_samples, dtype=torch.float32)
            dl = DataLoader(dssds, batch_size=bs, collate_fn=lambda x: x,
                            num_workers=n_threads, shuffle=False)
            write_ptr = 0

            for batch in tqdm(dl, total=ceil(n_samples / bs), disable=not verbose):
                inputs = batch['image'].to(device)

                if magnitude == 0:
                    output = batch['output'].to(device)
                else:
                    inputs.requires_grad_(True)
                    model._model.zero_grad()
                    output = model(inputs)
                    scores = torch.sum(sm(output / temperature, dim=1) ** 2, dim=1)
                    torch.log(torch.clamp(scores, min=1e-12)).sum().backward()
                    new_inputs = (inputs + magnitude * torch.sign(inputs.grad)).clamp(0, 1).detach()
                    inputs.requires_grad_(False)
                    model._model.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        output = model(new_inputs)

                scores = torch.sum(sm(output / temperature, dim=1) ** 2, dim=1)
                bsz = scores.shape[0]
                ds_scores[write_ptr:write_ptr + bsz] = scores.detach().cpu()
                write_ptr += bsz

            self._record(ds_key, ds_scores.reshape(-1), score_name)

        return self._df
