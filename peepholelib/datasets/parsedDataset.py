# General python stuff
from pathlib import Path as Path
from tqdm import tqdm
from math import ceil
import math


# tensordict
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT

# torch stuff
import torch
from torch.utils.data import DataLoader

# our stuff
from peepholelib.models.prediction_fns import multilabel_classification
from peepholelib.datasets.functional.results import results_one_hot_encoding

class ParsedDataset():
    def __init__(self, **kwargs):
        '''
        Creates instance of a parsed dataset.

        Args:
        - path (str): Dataset Path.

        '''
        self.path = Path(kwargs.get('path'))

        # computed in load_data()
        self._dss = None # this is the parsed datasets as PTD
        self._classes = None
        
        # used in the contexted manager
        is_contexted = None 
        return

    def get(self, ds_key, idx):
        return [self._dss[ds_key][idx]]
    
    # TODO: handle classes
    def get_classes(self):
        if not self._classes:
            raise RuntimeError('Data not loaded. Please run model.load_only() first.')

        return self._classes
    
    @classmethod
    def create_ds(cls, **kwargs):
        path = Path(kwargs['path'])
        ds_wrap = kwargs['ds_wrap']
        bs = kwargs.get('batch_size', 2**11)
        verbose = kwargs.get('verbose', False)
        ds_samplers = kwargs.get('ds_samplers', None)
        overwrite = kwargs.get('overwrite', False)

        path.mkdir(parents=True, exist_ok=True)
        cls_inst = cls(path=path)
        cls_inst._dss = {}

        ds_wrap.__load_data__(verbose=verbose)

        for ds_key in ds_wrap.__dataset__:
            ds_in_src = ds_wrap.__dataset__[ds_key]

            sampler = None
            if ds_samplers is not None:
                if ds_key in ds_samplers:
                    sampler = ds_samplers[ds_key]
                else:
                    split = ds_key.split("-")[-1]
                    if split in ds_samplers:
                        sampler = ds_samplers[split]

            if sampler is not None:
                perc = None
                if hasattr(sampler, "keywords") and sampler.keywords is not None:
                    perc = sampler.keywords.get("perc", None)
                if perc is None:
                    raise ValueError(f"Sampler for {ds_key} must be a partial with perc=... (got {sampler})")

                n_full = len(ds_in_src)
                n_keep = max(1, int(math.ceil(n_full * float(perc))))

                idx = torch.randperm(n_full)[:n_keep].tolist()
                ds_in_src = torch.utils.data.Subset(ds_in_src, idx)

                if verbose:
                    print(f"Subsampling {ds_key}: keeping {n_keep}/{n_full} (~{perc})")

            n_samples = len(ds_in_src)
            file_path = cls_inst.path / ('dss.' + ds_key)

            if overwrite and file_path.exists():
                file_path.unlink()

            if file_path.exists():
                if verbose:
                    print(f'File {file_path} exists. Loading from disk.')
                cls_inst._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
            else:
                if verbose:
                    print(f'Creating {ds_key} dataset with n_samples: ', n_samples)

                cls_inst._dss[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')

                sample = ds_in_src[0]
                if not isinstance(sample, dict):
                    raise TypeError(f"Dataset {ds_key} must return a dict. Got {type(sample)}")

                for key in sample.keys():
                    v = sample[key]
                    if not torch.is_tensor(v):
                        raise TypeError(f"Key '{key}' in {ds_key} must be a torch.Tensor. Got {type(v)}")

                    if verbose:
                        print(f'allocating {key} with shape {v.shape}')
                    cls_inst._dss[ds_key][key] = MMT.empty(
                        shape=torch.Size((n_samples,) + v.shape),
                        dtype=v.dtype
                    )

                ds_in = DataLoader(dataset=ds_in_src, batch_size=bs)
                ds_t = DataLoader(cls_inst._dss[ds_key], collate_fn=lambda x: x, batch_size=bs)

                for data_in, data_t in tqdm(zip(ds_in, ds_t), disable=not verbose, total=ceil(n_samples / bs)):
                    for key in data_in.keys():
                        data_t[key] = data_in[key]

            cls_inst._dss[ds_key].close()

        return cls_inst


    def parse_ds(self, **kwargs):
        model = kwargs['model']
        loaders = kwargs.get('loaders', None)
        bs = kwargs.get('batch_size', 2**11)
        verbose = kwargs.get('verbose', False)
        pred_fn = kwargs.get('pred_fn', multilabel_classification)
        result_fn = kwargs.get('result_fn', results_one_hot_encoding)

        if loaders is None:
            loaders = self._dss.keys()

        for ds_key in loaders:
            file_path = self.path / ('dss.' + ds_key)
            self._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            n_samples = len(self._dss[ds_key])

            sample = self._dss[ds_key][0:1]['image'].to(model.device)
            with torch.no_grad():
                out = model(sample)
            _out = out[0] if isinstance(out, (tuple, list)) else out

            if ('output' in self._dss[ds_key]) and ('pred' in self._dss[ds_key]) and ('result' in self._dss[ds_key]):
                continue

            self._dss[ds_key].batch_size = torch.Size((n_samples,))
            num_classes = _out.shape[1]

            self._dss[ds_key]['output'] = MMT.empty(shape=torch.Size((n_samples, num_classes)))
            self._dss[ds_key]['pred'] = MMT.empty(shape=torch.Size((n_samples,)))
            self._dss[ds_key]['result'] = MMT.empty(shape=torch.Size((n_samples,)))

            dl = DataLoader(self._dss[ds_key], collate_fn=lambda x: x, batch_size=bs)

            for data in tqdm(dl, disable=not verbose, total=ceil(n_samples / bs)):
                with torch.no_grad():
                    y = model(data['image'].to(model.device))
                y_predicted = y[0] if isinstance(y, (tuple, list)) else y

                predicted_labels = pred_fn(y_predicted).detach().cpu()
                data['output'] = y_predicted
                data['pred'] = predicted_labels
                data['result'] = result_fn(predicted_labels, data['label'])

            self._dss[ds_key].close()

        return self

    
    
    def load_only(self, **kwargs):
        '''
        Load already computed dataset.

        Args:
        - loaders (list[str]): load the specified loaders.
        - mode (str): Opens the file with the specified mode. See 'tensordict.PersistentTensorDict.from_h5()' for details. Defaults to 'r'.
        - verbose (bool): print progress messages.
        '''
        self.check_uncontexted()

        loaders = kwargs.get('loaders')
        mode = kwargs.get('mode', 'r')
        verbose = kwargs.get('verbose', False)

        self._dss = {}
        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting data from {ds_key}\n')
            
            # data file path
            _dfp = self.path/('dss.'+ds_key)

            if verbose: print(f'Loading files {_dfp} from disk. ')
            self._dss[ds_key] = PersistentTensorDict.from_h5(_dfp, mode=mode)

            _n_samples = len(self._dss[ds_key])
            if verbose: print('loaded n_samples: ', _n_samples)

        return
    
    def lazy_stack(self, **kwargs):
        '''
        Append other parsed datasets to self. `parsed datasets` contain then `self._dss` stribute.

        Args:
        others (list[peepholelib.datasets.dataset_base.DatasetBase]): list os datasets inheriting `DatasetBase` which have been parsed.
        '''
        others = kwargs.get('others')

        for ods in others:
            for ds_key in ods._dss:
                print('dskey: ', ds_key)
                if ds_key in self._dss:
                    raise RuntimeError(f'Trying to add {ds_key} from others, but key is already present in self.')
                    
                self._dss[ds_key] = ods._dss[ds_key]
                print(f'appending {ds_key}')
        return

    def __enter__(self):
        self._is_contexted = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        verbose = True 

        if self._dss == None:
            if verbose: print('no dss to close.')
        else:
            for ds_key in self._dss:
                if verbose: print(f'closing {ds_key}')
                self._dss[ds_key].close()

        self._is_contexted = False 
        return

    def check_uncontexted(self):
        if not self._is_contexted:
            raise RuntimeError('Function should be called within context manager')
        return


