# General python stuff
from pathlib import Path as Path
from tqdm import tqdm
from math import ceil


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
        self._is_contexted = None 
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
        '''
            Create datasets: creates a `ParsedDataset` instance containing a pointer for a PersistentTensorDict for each dataset, pointers are saved in a dictionary `cls._dss` whose keys are the `dataset_wraps`'s keys appendend with each loader insides its `__datasets__` dict. Generally the name of the PTDFs files are  `dss.<ds_wrap_key>-<loader>` and are saved in `path`.
            Returns: a pointer to the class istance `cls`.

            Args:
            - path (str): base path to store the datasets
            - dataset_wraps (dict[str, peepholelib.datasets.datasetWrap.DatasetWrap]): dict of dataset classes inhereting the `DatasetWrap` class. 
            - batch_size (int): batch size for processing.
            - n_threads (int): number of workers passed to DataLoaders.
            - ds_samplers (dict[str, function]): dict of functions to apply to the datasets (should have the same keys as the dataset_wraps)
            - verbose (bool): print progress messages.
        '''
        path = Path(kwargs['path'])
        dataset_wraps = kwargs['dataset_wraps'] 
        bs = kwargs.get('batch_size', 2**11)
        n_threads = kwargs.get('n_threads', 1)
        ds_samplers = kwargs.get('ds_samplers', None) 
        verbose = kwargs.get('verbose', False)

        path.mkdir(parents=True, exist_ok=True)
        cls_inst = cls(path=path)
        cls_inst._dss = {}

        if verbose: print(f'Creating datasets {list(dataset_wraps.keys())}. ')

        with cls_inst:
            for ds_name, ds_wrap in dataset_wraps.items():
                ds_wrap.__load_data__()

                if ds_samplers is not None:
                    ds_samplers[ds_name](ds=ds_wrap)

                for ds_key, ds_src in ds_wrap.__dataset__.items():

                    file_path = path / ('dss.' + ds_key)
                    if file_path.exists():
                        if verbose: print(f'File {file_path} exists. Skipping creation of {ds_key}.')
                        continue

                    n_samples = len(ds_src)
                    if verbose: print(f'Creating {ds_key} dataset with n_samples: ', n_samples)

                    cls_inst._dss[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
                    sample = ds_src[0]

                    for k, v in sample.items(): 
                        if verbose: print(f'allocating {k} with shape {v.shape}')
                        cls_inst._dss[ds_key][k] = MMT.empty(
                            shape=torch.Size((n_samples,) + v.shape),
                            dtype=v.dtype
                        )

                    dl_in = DataLoader(dataset=ds_src, batch_size=bs, num_workers=n_threads)
                    dl_t = DataLoader(cls_inst._dss[ds_key], collate_fn=lambda x: x, batch_size=bs)

                    for data_in, data_t in tqdm(zip(dl_in, dl_t), disable=not verbose, total=ceil(n_samples/bs)):
                        for key in data_in.keys():
                            data_t[key] = data_in[key]

        return cls_inst


    def parse_ds(self, **kwargs):
        '''
        Adds model outputs, predictions and results to an existing `ParsedDataset` dataset (for each loader).
        If these fields are already present, they are skipped.

        Args:
        - model (peepholelib.models.model_warp.ModelWrap): Model to use for predictions.
        - loaders (list[str]): list of dataset keys to parse. If None, all keys in self._dss are parsed.
        - batch_size (int): batch size for processing.
        - pred_fn (function): function to get predicted labels from model outputs. Defaults to `multilabel_classification`.
        - result_fn (function): function to get results from predicted labels and true labels. Defaults to `results_one_hot_encoding`.
        - verbose (bool): print progress messages.
        '''
        self.check_uncontexted()
        
        model = kwargs['model'] 
        loaders = kwargs.get('loaders', None)
        bs = kwargs.get('batch_size', 2**11)
        verbose = kwargs.get('verbose', False)
        pred_fn = kwargs.get('pred_fn', multilabel_classification) 
        result_fn = kwargs.get('result_fn', results_one_hot_encoding) 

        if loaders is None:
            loaders = self._dss.keys()

        # check if dataset is opened with the correct mode bofore any allocation/computation 
        r_modes = [ds_key for ds_key in loaders if self._dss[ds_key].mode == 'r']
        if len(r_modes) > 0:
            raise RuntimeError(f"Datasets '{r_modes}' are in 'r' mode. Make sure no other handle/process has it open read-only.") 

        for ds_key in loaders:
            file_path = self.path / ('dss.' + ds_key)

            if self._dss is None:
                self._dss = {}

            if ds_key not in self._dss:
                    self._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            if ('output' in self._dss[ds_key]) and ('pred' in self._dss[ds_key]) and ('result' in self._dss[ds_key]):
                continue

            n_samples = len(self._dss[ds_key])

            # Dry run to get shapes and dtype
            sample = self._dss[ds_key][0:1]['image'].to(model.device)
            sample_label = self._dss[ds_key][0:1]['label'].to(model.device)
            with torch.no_grad():
                _out = model(sample)

            self._dss[ds_key].batch_size = torch.Size((n_samples,))
            num_classes = _out.shape[1]
            _pred = pred_fn(_out)
            _res = result_fn(_pred, sample_label)

            # TODO: get the shapes from output, pred and result from the dryrun 
            self._dss[ds_key]['output'] = MMT.empty(shape=torch.Size((n_samples, num_classes)), dtype=_out.dtype)
            self._dss[ds_key]['pred'] = MMT.empty(shape=torch.Size((n_samples,)), dtype=_pred.dtype)
            self._dss[ds_key]['result'] = MMT.empty(shape=torch.Size((n_samples,)), dtype=_res.dtype)

            dl = DataLoader(self._dss[ds_key], collate_fn=lambda x: x, batch_size=bs)

            for data in tqdm(dl, disable=not verbose, total=ceil(n_samples / bs)):
                with torch.no_grad():
                    y_output = model(data['image'].to(model.device))

                predicted_labels = pred_fn(y_output).detach().cpu()
                data['output'] = y_output
                data['pred'] = predicted_labels
                data['result'] = result_fn(predicted_labels, data['label'])

        return 
    
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


