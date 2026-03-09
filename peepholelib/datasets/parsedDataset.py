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
    def parse_ds(cls, **kwargs):
        '''
        Parse datasets, saving images, labels, model output, 'result' (1 if samples are correctly classified, 0 otherwise). I know, copying images and labels is redundant, but it is convenient to have them all in a common structure for the downstream computations.
        Data is saved into a 'tensordict.PersistentTensorDict' at 'path/dss.<loader>', with 'loader' being the loaders keys (see peepholelib.datasets). Alreday existing files are skipped.
        Args:
        - path (str): base path to store parsed datasets.
        - dataset_wraps (dict{str: peepholelib.dataset_base.DatasetWrap}): Dictionary with key being the name, and value an instance of specific dataset inheriting `datasets.DatasetWrap`.
        - ds_samplers (dict(str: dict())): Dictionary with same keys as `datasets`, and values being a sampler (see `datasets.functional.samplers`). Facultative.
        - keys_to_copy (dict(str: list[str])): Dictionary with same keys as `datasets`, and values lists of keys to copy from the dataset_wraps. Skips already present keys. If `None` copies all keys (which are not already present). Defaults to `None`. 
        - inference_fn (callable): Inference function that returns a dictionary of outputs to be saved with the parsed dataset. This is useful if the model does not return a dictionary or to add extra computation to its outputs, e.g. One might pass a function which returns just `image` and `label`, so the parsed dataset can be used for training a model; another example is to add `result` and `output` for havin the model's logits or a correct classification. Defaults to `None`

        - batch_size (int): Creates dataloader to do computation in batch size. Defaults to 64.
        - n_threads (int): 'num_workers' passed to 'torch.utils.data.DataLoader'. Defaults to 1.
        - verbose (bool): print progress messages.
        '''
        
        path = Path(kwargs.get('path'))
        ds_wraps = kwargs.get('dataset_wraps')
        ds_samplers = kwargs.get('ds_samplers', None)
        keys_to_copy = kwargs.get('keys_to_copy', None)
        inference_fn = kwargs.get('inference_fn', None)
        bs = kwargs.get('batch_size', 64) 
        n_threads = kwargs.get('n_threads', 1) 
        verbose = kwargs.get('verbose', False) 

        path.mkdir(parents=True, exist_ok=True)
        cls_inst = cls(path = path)
        cls_inst._dss = {}

        # enter the context manager
        with cls_inst:
            for ds_name, ds_wrap in ds_wraps.items():

                ds_wrap.__load_data__()

                if ds_samplers != None and ds_name in ds_samplers:
                    if verbose: print(f'Applying {ds_samplers[ds_name]} to {ds_name}')
                    ds_samplers[ds_name](ds = ds_wrap)

                for ds_key in ds_wrap.__dataset__:
                    if verbose: print(f'\n ---- Getting data from {ds_key}\n')
                    file_path = cls_inst.path/('dss.'+ds_key)
                    
                    if file_path.exists():
                        if verbose: print(f'File {file_path} exists. Loading from disk.')

                        cls_inst._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
                        n_samples = len(cls_inst._dss[ds_key])
                        # this is a workaround for when loading PTDs with already populated MMTs
                        cls_inst._dss[ds_key].batch_size = torch.Size((n_samples,))

                        _ns_wrap = len(ds_wrap.__dataset__[ds_key])

                        # Check if PTD's number of samples is the same ds_wrap's 
                        if n_samples != _ns_wrap:
                            raise RuntimeError(f'Dataset Wrap {ds_key} has {_ns_wrap} samples, but the parsed one has {n_samples} samples. Something is wrong here.')
                        
                        if verbose: print('loaded n_samples: ', n_samples)
                    else:
                        n_samples = len(ds_wrap.__dataset__[ds_key])

                        if verbose: print('Creating dataset with n_samples: ', n_samples)

                        cls_inst._dss[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')

                    #------------------------
                    # Pre-allocation 
                    #------------------------
                    # dry run to get shapes
                    sample = next(iter(DataLoader(
                            dataset = ds_wrap.__dataset__[ds_key],
                            batch_size = 1,
                            shuffle = False
                            ))) 

                    # check is all keys_to_copy are withing the samples
                    if keys_to_copy == None:
                        keys_to_copy = list(sample.keys())
                    elif len(list(set(keys_to_copy)-set(sample.keys()))) > 0:
                           raise RuntimeError(f'keys_to_copy {keys_to_copy} should be a subset of the keys from a ds_wrap sample, but {ds_key} has {list(sample.keys())}.')
                    
                    # only copy the keys that are not already within the PTD
                    in_ktc  = list(set(keys_to_copy) - set(cls_inst._dss[ds_key].keys()))

                    if verbose: print(f'New keys to copy from dataset wrap: {in_ktc}')

                    for key in in_ktc:
                        _v = sample[key]
                        _shape = (n_samples,)+_v.shape[1:]

                        if verbose: print(f'Allocating {key} with shape {_shape}')

                        cls_inst._dss[ds_key][key] = MMT.empty(
                                shape = torch.Size(_shape),
                                dtype = _v.dtype
                                )

                    if inference_fn != None:
                        # make the inference
                        with torch.no_grad():
                            _res = inference_fn(data = sample)

                        out_ktc = list(set(_res.keys())-set(cls_inst._dss[ds_key].keys()))
                        if verbose: print(f'New output keys to add: {out_ktc}')
                         
                        for key in out_ktc:
                            _v = _res[key]
                            _shape = (n_samples,)+_v.shape[1:]

                            if verbose: print(f'Allocating {key} with shape {_shape}')

                            cls_inst._dss[ds_key][key] = MMT.empty(
                                    shape = torch.Size(_shape),
                                    dtype = _v.dtype
                                    )

                    # Close PTD create with mode 'w' and re-open it with mode 'r+'
                    # This is done so we can use multiple workers with the dataloaders 
                    cls_inst._dss[ds_key].close()
                    cls_inst._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

                    #------------------------
                    # copy images and labels
                    #------------------------
                    # create dataloader of input dataset
                    dl_ori = DataLoader(
                            dataset = ds_wrap.__dataset__[ds_key],
                            batch_size = bs,
                            shuffle = False
                            ) 

                    dl_dst = DataLoader(
                            cls_inst._dss[ds_key],
                            batch_size = bs,
                            collate_fn = lambda x:x,
                            shuffle = False,
                            num_workers = n_threads
                            )
                    
                    if len(in_ktc) == 0 and len(out_ktc) == 0: 
                        if verbose: print(f'Nothing to parse. Skipping.')
                        continue

                    if verbose: print(f'Parsing {ds_key}')
                    for data_in, data_t in tqdm(zip(dl_ori, dl_dst), disable=not verbose, total=ceil(n_samples/bs)): 
                        # parse input ds
                        for key in in_ktc:
                            data_t[key] = data_in[key]
                        
                        # parse outputs 
                        if inference_fn != None:
                            with torch.no_grad():
                                _res = inference_fn(data = data_in)

                            for key in out_ktc:
                                data_t[key] = _res[key]

        return cls_inst

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
        verbose = kwargs.get('verbose', True)

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
