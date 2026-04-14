# General python stuff
from pathlib import Path as Path
from tqdm import tqdm
from math import ceil

# tensordict
from tensordict import PersistentTensorDict
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

# torch stuff
import torch
from torch.utils.data import DataLoader, Subset

class _ShardedPTD:
    '''
    Read-only wrapper over multiple `PersistentTensorDict` shards that presents them as a single dataset.
    Shard files are named `dss.<ds_key>.chunk_<i>`.
    '''
    def __init__(self, shards):
        self.shards = shards
        lengths = [len(s) for s in shards]
        self._cum = [0]
        for l in lengths:
            self._cum.append(self._cum[-1] + l)
        self._total = self._cum[-1]

    def _resolve(self, idx):
        if idx < 0:
            idx = self._total + idx
        for i in range(len(self.shards)):
            if idx < self._cum[i + 1]:
                return i, idx - self._cum[i]
        raise IndexError(f'Index {idx} out of range for sharded dataset of size {self._total}')

    def __len__(self):
        return self._total

    def keys(self):
        return self.shards[0].keys()

    def __getitem__(self, idx):

        if isinstance(idx, int):
            si, li = self._resolve(idx)
            return self.shards[si][li]

        if isinstance(idx, slice):    
            idx = list(range(*idx.indices(self._total)))

        # list / tensor of indices — collect per key then stack
        indices = idx if isinstance(idx, list) else list(idx)
        ks = list(self.keys())
        per_key = {k: [] for k in ks}
        for global_i in indices:
            si, li = self._resolve(int(global_i))
            sample = self.shards[si][li]
            for k in ks:
                per_key[k].append(sample[k])
        return TensorDict(
            {k: torch.stack(v) for k, v in per_key.items()},
            batch_size=len(indices)
        )
    
    def __getitems__(self, indices):
        ks = list(self.keys())

        # group indices by shard to allow batch reads
        shard_groups = {}  # si -> list of (result_position, local_idx)
        for pos, global_i in enumerate(indices):
            si, li = self._resolve(int(global_i))
            if si not in shard_groups:
                shard_groups[si] = []
            shard_groups[si].append((pos, li))

        # pre-allocate result slots
        per_key = {k: [None] * len(indices) for k in ks}
        for si, pos_local_pairs in shard_groups.items():
            positions, local_idxs = zip(*pos_local_pairs)
            batch = self.shards[si][list(local_idxs)]  # one batched read per shard
            for k in ks:
                for pos, val in zip(positions, batch[k]):
                    per_key[k][pos] = val

        return TensorDict(
            {k: torch.stack(v) for k, v in per_key.items()},
            batch_size=len(indices)
        )

    def close(self):
        for s in self.shards:
            s.close()

class _StackedDS:
    '''
    This class is a wrap for multiple `PersistentTensorDicts` (PTDs) with DIFFERENT KEYS, oferring a common interface to emulate as if it was one PTD with all the keys. 
    '''
    def __init__(self, **kwargs):
        '''
        Args:
        - ori (tensordict.PersistentTensorDict | peepholelib.datasets.datasetWrap.DatasetWrap): Original dataset wrapped or a PTD. Both are supported since sampling from both return a dictionary.
        '''
        self.ori = kwargs['ori']
        self.transform = lambda x: x

        # set in stack_inference
        self.inf = None

        s_ori = self.ori[0]

        self.k_ori = list(s_ori.keys())
        self.k_inf = [] 
        self.len = len(self.ori)
        return
    
    def set_transform(self, transform=None):
        if transform != None:
            self.transform = transform
        return

    def stack_inference(self, inf):
        s_ori = self.ori[0]
        s_inf = inf[0]

        if len(self.ori) != len(inf):
            raise RuntimeError(f'Original dataset has {len(self.ori)} samples, parsed inference values have {len(inf)} samples. They should have the same number of samples.')

        # if keys appear in both, we ignore it from the original
        self.k_ori = list(set(s_ori.keys()) - set(s_inf.keys())) 
        self.k_inf = list(s_inf.keys()) 

        # save the inf
        self.inf = inf

        return

    def __getitem__(self, idx):
        r = {}

        for k in self.k_ori:
            r[k] = self.ori[idx][k] 

        if self.inf != None:
            for k in self.k_inf:
                r[k] = self.inf[idx][k] 

        return self.transform(r) 
    
    def __getitems__(self, idx):
        r = TensorDict({}, batch_size=len(idx))

        ori_batch = self.ori.__getitems__(idx)
        for k in self.k_ori:
            r[k] = ori_batch[k]

        if self.inf != None:
            inf_batch = self.inf.__getitems__(idx)
            for k in self.k_inf:
                r[k] = inf_batch[k]

        return self.transform(r)

    def __len__(self):
        return self.len

    def keys(self):
        return self.k_ori + self.k_inf

    def close(self):
        if self.inf != None:
            self.inf.close()

        if hasattr(self.ori, 'close'):
            self.ori.close()

class ParsedDataset():
    def __init__(self, **kwargs):
        '''
        Creates instance of a parsed dataset.

        Args:
        - path (str): Dataset Path.

        '''
        self.path = Path(kwargs.get('path'))

        # set in parsed_dataset(), parse_inference(), load_only()
        self._dss = {} # this is the parsed datasets as PTD
        self._dss_ori = {} 

        # used in the contexted manager
        self._is_contexted = None
        return

    def get(self, ds_key, idx):
        return [self._dss[ds_key][idx]]
   
    def parse_dataset(self, **kwargs):
        '''
        Parse datasets, saving original values from the dataset on a `tensordict.PersistentTensorDict` at 'self.path/dss.<loader>', with 'loader' being the loaders keys (see peepholelib.datasets). The values to copy can be defined using the `keys_to_copy` argument, otherwise copies all keys for each `datasetWrap`, skipping already existing values.

        When `chunk_size` is set the dataset is split into multiple shard files named `dss.<loader>.chunk_<i>` and wrapped in a `_ShardedPTD`. This keeps individual files small and avoids memory pressure on large datasets. `load_only()` detects these shard files automatically.

        Args:
        - dataset_wraps (dict{str: peepholelib.dataset_base.DatasetWrap}): Dictionary with key being the name, and value an instance of specific dataset inheriting `datasets.DatasetWrap`.
        - ds_samplers (dict(str: dict())): Dictionary with same keys as `datasets`, and values being a sampler (see `datasets.functional.samplers`). Facultative.
        - keys_to_copy (dict(str: list[str])): Dictionary with same keys as `datasets`, and values lists of keys to copy from the dataset_wraps. Skips already present keys. If `None` copies all keys (which are not already present). Defaults to `None`.
        - chunk_size (int | None): If set, split each loader into shards of at most `chunk_size` samples saved as separate files. If `None` (default) a single file is written as before.
        - batch_size (int): Creates dataloader to do computation in batch size. Defaults to 64.
        - n_threads (int): 'num_workers' passed to 'torch.utils.data.DataLoader'. Defaults to 1.
        - verbose (bool): print progress messages.
        '''
        self.check_uncontexted()

        ds_wraps = kwargs.get('dataset_wraps')
        ds_samplers = kwargs.get('ds_samplers', None)
        keys_to_copy = kwargs.get('keys_to_copy', None)
        chunk_size = kwargs.get('chunk_size', None)
        bs = kwargs.get('batch_size', 64)
        n_threads = kwargs.get('n_threads', 1)
        verbose = kwargs.get('verbose', False)

        self.path.mkdir(parents=True, exist_ok=True)

        # be sure that the datasets do not have a transform.
        # transforms should be set in `self.parse_inferece()`
        has_trans = []
        for ds_name, ds_wrap in ds_wraps.items():
            if ds_wrap.has_transforms:
                has_trans.append(ds_name)
        if len(has_trans) > 0:
            raise RuntimeError(f'Found `transforms` within the given `dataset_wraps`: {has_trans}. DatasetWraps are expected to not have transforms at this point since they will be set in `parse_inferece()`')


        # enter the context manager
        for ds_name, ds_wrap in ds_wraps.items():
            ds_wrap.__load_data__()

            if ds_samplers != None and ds_name in ds_samplers:
                if verbose: print(f'Applying sampler to {ds_name}')
                ds_samplers[ds_name](ds = ds_wrap)

            for ds_key in ds_wrap.__dataset__.keys():
                if verbose: print(f'\n ---- Getting data from {ds_key}\n')
                n_samples = len(ds_wrap.__dataset__[ds_key])

                # sample for dry run to get shapes
                sample = next(iter(DataLoader(
                        dataset = ds_wrap.__dataset__[ds_key],
                        batch_size = 1,
                        shuffle = False
                        )))

                # check / resolve keys_to_copy
                if keys_to_copy == None:
                    _ktc = list(sample.keys())
                elif len(list(set(keys_to_copy)-set(sample.keys()))) > 0:
                    raise RuntimeError(f'keys_to_copy {keys_to_copy} should be a subset of the keys from a ds_wrap sample, but {ds_key} has {list(sample.keys())}.')
                else:
                    _ktc = keys_to_copy

                effective_chunk_size = chunk_size if chunk_size is not None else n_samples
                n_chunks = ceil(n_samples / effective_chunk_size)

                if verbose: print(f'Splitting {ds_key} into {n_chunks} chunks of up to {effective_chunk_size} samples')

                shards = []
                for chunk_i in range(n_chunks):
                    chunk_start = chunk_i * effective_chunk_size
                    chunk_end = min(chunk_start + effective_chunk_size, n_samples)
                    chunk_n = chunk_end - chunk_start
                    chunk_path = self.path / f'dss.{ds_key}' / f'chunk_{chunk_i}'

                    chunk_path.parent.mkdir(parents=True, exist_ok=True)
                    if chunk_path.exists():
                        if verbose: print(f'Chunk {chunk_i}: {chunk_path} exists. Loading from disk.')
                        ptd = PersistentTensorDict.from_h5(chunk_path, mode='r+')
                        ptd.batch_size = torch.Size((chunk_n,))
                    else:
                        if verbose: print(f'  Chunk {chunk_i}: Creating ({chunk_n} samples).')
                        ptd = PersistentTensorDict(filename=chunk_path, batch_size=[chunk_n], mode='w')

                    in_ktc = list(set(_ktc) - set(ptd.keys()))

                    if len(in_ktc) > 0:
                        for key in in_ktc:
                            if verbose: print(f'Allocating {key}')
                            ptd[key] = MMT.empty(
                                    shape = torch.Size((chunk_n,) + sample[key].shape[1:]),
                                    dtype = sample[key].dtype
                                    )

                        ptd.close()
                        ptd = PersistentTensorDict.from_h5(chunk_path, mode='r+')

                        subset = Subset(ds_wrap.__dataset__[ds_key], range(chunk_start, chunk_end))

                        dl_ori = DataLoader(
                                dataset = subset,
                                batch_size = bs,
                                shuffle = False
                                )

                        dl_dst = DataLoader(
                                ptd,
                                batch_size = bs,
                                collate_fn = lambda x: x,
                                shuffle = False,
                                num_workers = n_threads
                                )

                        if verbose: print(f'Parsing chunk {chunk_i}')
                        for data_in, data_t in tqdm(zip(dl_ori, dl_dst), disable=not verbose, total=ceil(chunk_n/bs)):
                            for key in in_ktc:
                                data_t[key] = data_in[key]

                    shards.append(ptd)

                self._dss[ds_key] = _StackedDS(ori=_ShardedPTD(shards))
        return
    
    def parse_inference(self, **kwargs):
        '''
        Parse inference results, e.g., output, `result` (1 if samples are correctly classified, 0 otherwise) into `tensordict.PersistentTensorDict`s at `path/dss.<loader>.<name>`, with 'loader' being the loaders keys (see `self._dss.keys()` in `self.parse_dataset()`) and `name` is the key from the dictionary passed in the `inference_fns` argument. Already existing keys are skipped.

        Args:
        - loaders (list[str]): list of keys from `self._dss` to apply the inference. If `None` apply the inference for all keys within `self._dss.keys()`. Defaults to `None`. 
        - inference_fns (dict{str: callable}): Inference functions that return a dictionary of outputs to be saved with the parsed dataset. This is useful if the model does not return a dictionary or to add extra computation to its outputs.
        - transforms (dict{str: callable}): Dictionary with keys matching the loaders and transforms as values. A transorm takes as input a sample from the parsed dataset (`self._dss`) and edits its values. If `None` or in case of missing keys, uses `lambda x: x`. Defaults to `None`. 
        - batch_size (int): Creates dataloader to do computation in batch size. Defaults to 64.
        - n_threads (int): 'num_workers' passed to 'torch.utils.data.DataLoader'. Defaults to 1.
        - verbose (bool): print progress messages.
        '''
        
        loaders = kwargs.get('loaders', None)
        inference_fns = kwargs['inference_fns']
        transforms = kwargs.get('transforms', None)
        bs = kwargs.get('batch_size', 64) 
        n_threads = kwargs.get('n_threads', 1) 
        verbose = kwargs.get('verbose', False) 

        if loaders == None: loaders = list(self._dss.keys())

        for ds_key in loaders:
            # save the pointers for the original stackedDSs
            if (not (ds_key in self._dss_ori.keys())) and (ds_key in self._dss.keys()):
                self._dss_ori[ds_key] = self._dss.pop(ds_key)

            for inf_name, inf_fn in inference_fns.items():
                if verbose: print(f'\n ---- Getting data from {ds_key}\n')
                file_path = self.path/f'dss.{ds_key}.{inf_name}' 
                inf_ds_key = ds_key + '-' + inf_name
             
                # create new StackedDS copying the pointer to the original parsed DS
                # keys in self._dss are altered
                self._dss[inf_ds_key] = _StackedDS(ori=self._dss_ori[ds_key].ori)
                self._dss[inf_ds_key].set_transform(transforms[ds_key] if transforms != None and ds_key in transforms else None)
                
                if file_path.exists():
                    if verbose: print(f'File {file_path} exists. Loading from disk.')

                    ptd = PersistentTensorDict.from_h5(file_path, mode='r+')
                    n_samples = len(ptd)
                    # this is a workaround for when loading PTDs with already populated MMTs
                    ptd.batch_size = torch.Size((n_samples,))
                    
                    if verbose: print('loaded n_samples: ', n_samples)
                else:
                    n_samples = len(self._dss[inf_ds_key])
                    if verbose: print('Creating dataset with n_samples: ', n_samples)
                    ptd = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')

                # sample for dry run to get shapes
                sample = self._dss[inf_ds_key][0:1]
                with torch.no_grad():
                    _res = inf_fn(data = sample)

                out_ktc = list(set(_res.keys())-set(ptd.keys()))
                
                if len(out_ktc) > 0:
                    for key in out_ktc:
                        if verbose: print(f'Allocating {key}')
                        ptd[key] = MMT.empty(
                                shape = torch.Size((n_samples,)+_res[key].shape[1:]),
                                dtype = _res[key].dtype
                                )

                    # Close PTD create with mode 'w' and re-open it with mode 'r+'
                    # This is done so we can use multiple workers with the dataloaders 
                    ptd.close()
                    ptd = PersistentTensorDict.from_h5(file_path, mode='r+')

                    #------------------------
                    # copy images and labels
                    #------------------------
                    dl_ori = DataLoader(
                            dataset = self._dss[inf_ds_key],
                            batch_size = bs,
                            collate_fn = lambda x:x,
                            shuffle = False,
                            num_workers = n_threads
                            )

                    dl_dst = DataLoader(
                            dataset = ptd,
                            batch_size = bs,
                            collate_fn = lambda x:x,
                            shuffle = False,
                            num_workers = n_threads
                            )
                    
                    if verbose: print(f'Parsing inference for {inf_ds_key}')
                    with torch.no_grad():
                        for data_in, data_t in tqdm(zip(dl_ori, dl_dst), disable=not verbose, total=ceil(n_samples/bs)):
                            _res = inf_fn(data = data_in)
                            for key in out_ktc:
                                data_t[key] = _res[key]

                self._dss[inf_ds_key].stack_inference(inf=ptd)
        return 
    
    def load_only(self, **kwargs):
        '''
        Load already parsed dataset, sets transforms, and load inference values. Parsed datasets are saved on `self._dss[<loader>]` for each `loader` within `loaders`. If `inference_names` is passed, the function backs up the parsed dataset in `self._dss_ori`, and instead saves `self._dss[<loader>-<inf_name>] = _StackedDS(ori=self._dss[<loader>])` for each `inf_name` in `inference_names[<loader>]`, stacking the respective inference values. As such, all inferences will point to the same original parsed dataset (the one computed with `self.parse_dataset()`). 

        Args:
        - loaders (list[str]): load the specified loaders.
        - transforms (dict{str: callable}): Dictionary with keys matching `loaders` and callable transforms as values (e.g., `peepholelib.datasets.functional.transforms.TransformWrap`). Transforms should be same as the ones used in `self.parse_inference()`. A transorm takes as input a sample from the parsed dataset (`self._dss`) and edits its values. If `None`, uses `lambda x: x`. Defaults to `None`. 
        - inference_names (dict{str:list[str]}): Inference names given as the keys of `inference_fns` to `self.parse_inference()`. Empty lists (`[]`) will result in no inference for a given `loader`. If `None` no inference values are loaded for all `loaders`. Defaults to `None`. 
        - mode (str): Opens the file with the specified mode. See 'tensordict.PersistentTensorDict.from_h5()' for details. Defaults to 'r'.
        - verbose (bool): print progress messages.
        '''
        self.check_uncontexted()

        loaders = kwargs.get('loaders')
        transforms = kwargs.get('transforms', None)
        inf_names = kwargs.get('inference_names', None)
        mode = kwargs.get('mode', 'r')
        verbose = kwargs.get('verbose', True)

        # TODO: Should we try to close any eventually loaded PTD?
        self.__exit__(None, None, None)
        self._dss = {}
        self._dss_ori = {}

        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting data from {ds_key}\n')

            # detect sharded vs single-file layout
            chunk_paths = sorted((self.path / f'dss.{ds_key}').glob('chunk_*'))
            if verbose: print(f'Loading {len(chunk_paths)} shards for {ds_key}.')
            shards = []
            for cp in chunk_paths:
                if verbose: print(f'  Loading {cp}')
                _ptd = PersistentTensorDict.from_h5(cp, mode=mode)
                _ptd.batch_size = torch.Size((len(_ptd),))
                shards.append(_ptd)
            ori = _ShardedPTD(shards)
            
            self._dss[ds_key] = _StackedDS(ori=ori)

            if inf_names == None:
                self._dss[ds_key].set_transform(transforms[ds_key] if transforms != None else None)
                _n_samples = len(self._dss[ds_key])

            else: 
                if len(inf_names[ds_key]) == 0:
                    self._dss[ds_key].set_transform(transforms[ds_key] if transforms != None else None)
                    _n_samples = len(self._dss[ds_key])
                else:
                    for inf_name in inf_names[ds_key]:
                        # back up the pointer to the original ds
                        if (not (ds_key in self._dss_ori.keys())) and (ds_key in self._dss.keys()):
                            self._dss_ori[ds_key] = self._dss.pop(ds_key)
                        
                        inf_ds_key = ds_key + '-' + inf_name

                        # create new StackedDS copying the pointer to the original parsed DS
                        self._dss[inf_ds_key] = _StackedDS(ori=self._dss_ori[ds_key].ori)
                        self._dss[inf_ds_key].set_transform(transforms[ds_key] if transforms != None and ds_key in transforms else None)

                        # stack inference values — single H5 file
                        inf_path = self.path / f'dss.{ds_key}.{inf_name}'
                        if not inf_path.exists():
                            raise FileNotFoundError(
                                f'No inference data found for {inf_ds_key}. '
                                f'Expected a single H5 file at {inf_path}.'
                            )
                        if verbose: print(f'Loading inference for {inf_ds_key} from {inf_path}.')
                        _td = PersistentTensorDict.from_h5(inf_path, mode=mode)
                        _td.batch_size = torch.Size((len(_td),))

                        self._dss[inf_ds_key].stack_inference(inf=_td)

                    _n_samples = len(self._dss[inf_ds_key])
            if verbose: print('loaded n_samples: ', _n_samples)
        return
    
    def __enter__(self):
        self._is_contexted = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        verbose = True 

        for ds_key in self._dss_ori:
            if verbose: print(f'closing {ds_key}')
            self._dss_ori[ds_key].close()

        for ds_key in self._dss:
            if verbose: print(f'closing {ds_key}')
            self._dss[ds_key].close()

        self._is_contexted = False 
        return

    def check_uncontexted(self):
        if not self._is_contexted:
            raise RuntimeError('Function should be called within context manager')
        return

