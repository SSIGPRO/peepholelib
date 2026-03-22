from tqdm import tqdm

import torch
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

from peepholelib.models.model_wrap import get_in_activations

def get_coreVectors_pred(self, **kwargs):
    '''
    Like get_coreVectors, but also supports reducers used in prediction-time pipelines.

    For loaders listed in `fit_loaders`, labels are passed to the reducer.
    For all other loaders, predictions are passed instead.

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): Parsed datasets.
    - fit_loaders (list[str]): Loaders whose data was used to fit the reducer. Labels are passed for these.
    - input_key (str): Key for model input in dataset batches. Defaults to 'image'.
    - label_key (str): Key for labels in dataset batches. Defaults to 'label'.
    - pred_key (str): Key for predictions in dataset batches. Defaults to 'pred'.
    - reducers (dict[str, Callable]): Per-module dimensionality reduction callables.
    - activations_parser (callable): Parser for model activations. Defaults to get_in_activations.
    - batch_size (int): Batch size. Defaults to 64.
    - n_threads (int): DataLoader workers. Defaults to 1.
    - verbose (bool): Print progress. Defaults to False.
    '''
    self.check_uncontexted()

    datasets = kwargs.get('datasets')
    fit_loaders = kwargs.get('fit_loaders', [])
    input_key = kwargs.get('input_key', 'image')
    label_key = kwargs.get('label_key', 'label')
    pred_key = kwargs.get('pred_key', 'pred')
    reducers = kwargs.get('reducers')
    activations_parser = kwargs.get('activations_parser', get_in_activations)
    bs = kwargs.get('batch_size', 64)
    n_threads = kwargs.get('n_threads', 1)
    save_input = kwargs.get('save_input', True)
    save_output = kwargs.get('save_output', False)
    verbose = kwargs.get('verbose', False)

    model = self._model
    device = self._model.device

    if reducers.keys() != model._target_modules.keys():
        raise RuntimeError(
            f'Keys inconsistency between reducers and target_modules\n'
            f'reducers keys: {reducers.keys()}\ntarget_modules: {model._target_modules.keys()}'
        )

    model.set_activations(save_input=save_input, save_output=save_output)

    self._corevds = {}
    for ds_key in datasets._dss:
        if verbose: print(f'\n ---- Getting core vectors for {ds_key}\n')
        file_path = self.path / (self.name + '.' + ds_key)

        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._corevds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
            n_samples = len(self._corevds[ds_key])
            if verbose: print('loaded n_samples: ', n_samples)
            self._corevds[ds_key].batch_size = torch.Size((n_samples,))
        else:
            n_samples = len(datasets._dss[ds_key])
            self._corevds[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
            if verbose: print('created corevectors with n_samples: ', n_samples)

        _modules_to_save = []

        with torch.no_grad():
            model(datasets._dss[ds_key][0:1][input_key].to(device))
            _act0 = activations_parser(model._acts)

        for mk in model._target_modules.keys():
            if not (mk in self._corevds[ds_key]):
                _cv = reducers[mk](act_data=_act0[mk], labels=datasets._dss[ds_key][0:1]['label'])
                cv_shape = _cv.shape[1:]
                if verbose: print('allocating core vectors for module: ', mk)
                self._corevds[ds_key][mk] = MMT.empty(shape=((n_samples,) + cv_shape), dtype=_cv.dtype)
                _modules_to_save.append(mk)

        if verbose: print('Modules to save: ', _modules_to_save)

        self._corevds[ds_key].close()
        self._corevds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

        if len(_modules_to_save) == 0:
            print(f'No new core vectors for {ds_key}, skipping')
            continue

        if verbose: print(f'\n ---- Getting corevectors for {ds_key}\n')

        use_labels = ds_key in fit_loaders

        ds_dl = DataLoader(datasets._dss[ds_key], batch_size=bs, collate_fn=lambda x: x, shuffle=False, num_workers=n_threads)
        cvs_dl = DataLoader(self._corevds[ds_key], batch_size=bs, collate_fn=lambda x: x, shuffle=False, num_workers=n_threads)

        with torch.no_grad():
            for cvs_data, ds_data in tqdm(zip(cvs_dl, ds_dl), disable=not verbose, total=len(cvs_dl)):
                model(ds_data[input_key].to(device))
                class_refs = ds_data[label_key] if use_labels else ds_data[pred_key]
                for mk in _modules_to_save:
                    act_data = activations_parser(model._acts)
                    cvs_data[mk] = reducers[mk](act_data=act_data[mk], labels=class_refs)

    model.set_activations(save_input=False, save_output=False)

    return
