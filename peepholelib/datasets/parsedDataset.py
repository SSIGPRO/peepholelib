# General python stuff
from pathlib import Path as Path
from tqdm import tqdm
from math import ceil

# tensordict
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT

# torch stuff
import torch
from torch.utils.data import DataLoader, Subset

# Our stuff
from peepholelib.utils.ptd_wraps import _ShardedPTD, _StackedDS 

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
<<<<<<< HEAD
        - dataset_wraps (dict{str: peepholelib.dataset_base.DatasetWrap}): Dictionary with key being the name, and value an instance of specific dataset inheriting `datasets.DatasetWrap`.
=======
        - datasets (peepholelib.dataset_base.DatasetWrap): Dictionary with key being the name, and value an instance of specific dataset inheriting `datasets.DatasetWrap`.
        - ds_parsers (dict(str: callable)): Dictionary with same keys as `datasets`, and values being functions taking batched dataset samples and parsing into a dictionary with keys = ['images', 'labels']. 
>>>>>>> 0eef6bb (implement svg kernel svd (#127))
        - ds_samplers (dict(str: dict())): Dictionary with same keys as `datasets`, and values being a sampler (see `datasets.functional.samplers`). Facultative.
        - keys_to_copy (list[str]): List with same keys as `datasets`, and values lists of keys to copy from the dataset_wraps. Skips already present keys. If `None` copies all keys (which are not already present). Defaults to `None`.
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

        for ds_name, ds_wrap in ds_wraps.items():
            ds_wrap.__load_data__()

            _sampler_applied = False
            for ds_key in ds_wrap.__dataset__.keys():
                if verbose: print(f'\n ---- Getting data from {ds_key}\n')
                n_samples = len(ds_wrap.__dataset__[ds_key])
                _chunk_size = chunk_size if chunk_size is not None else n_samples
                n_chunks = ceil(n_samples/_chunk_size)

                sample = next(iter(DataLoader(dataset=ds_wrap.__dataset__[ds_key], batch_size=1, shuffle=False)))
                if keys_to_copy is None:
                    _ktc = list(sample.keys())
                else:
                    _ktc = keys_to_copy

                ds_folder = self.path/f'dss.{ds_key}'
                shards = []
                if ds_folder.exists():
                    if verbose: print(f'Loading {ds_key} from disk.')
                    for chunk_path in sorted(ds_folder.glob('chunk_*'), key=lambda p: int(p.name.split('_')[-1])):
                        ptd = PersistentTensorDict.from_h5(chunk_path, mode='r')
                        ptd.batch_size = torch.Size((len(ptd),))
                        shards.append(ptd)
                else:
                    if not _sampler_applied and ds_samplers is not None and ds_name in ds_samplers:
                        # TODO: parse indexes to re-enable this
                        # apply the sampler only if it is a new parsing. This sort of defeat the purpose of keys to copy
                        if verbose: print(f'Applying sampler to {ds_name}')
                        ds_samplers[ds_name](ds = ds_wrap)
                        _sampler_applied = True

                        n_samples = len(ds_wrap.__dataset__[ds_key])
                        _chunk_size = chunk_size if chunk_size is not None else n_samples
                        n_chunks = ceil(n_samples/_chunk_size)

                    ds_folder.mkdir(parents=True, exist_ok=True)
                    for chunk_i in range(n_chunks):
                        chunk_start = chunk_i * _chunk_size
                        chunk_end = min(chunk_start + _chunk_size, n_samples)
                        chunk_n = chunk_end - chunk_start
                        chunk_path = ds_folder / f'chunk_{chunk_i}'

                        if verbose: print(f' - Chunk {chunk_i}: Creating ({chunk_n} samples).')

                        ptd = PersistentTensorDict(filename=chunk_path, batch_size=[chunk_n], mode='w')
                        ptd.close()
                        ptd = PersistentTensorDict.from_h5(chunk_path, mode='r+')
                        ptd.batch_size = torch.Size((chunk_n,))

                        for key in _ktc:
                            shape = torch.Size((chunk_n,)+sample[key].shape[1:])
                            if verbose: print(f' - Allocating {key} with shape {shape}.')
                            ptd[key] = MMT.empty(
                                shape = shape,
                                dtype = sample[key].dtype
                                )

                        dl_ori = DataLoader(
                            dataset = Subset(
                                ds_wrap.__dataset__[ds_key],
                                range(chunk_start, chunk_end)
                                ),
                            batch_size = bs,
                            shuffle = False
                            )

                        dl_dst = DataLoader(
                                ptd,
                                batch_size=bs,
                                collate_fn=lambda x: x,
                                shuffle=False,
                                num_workers=n_threads
                                )

                        if verbose: print(f' - Parsing chunk {chunk_i}')
                        for data_in, data_t in tqdm(zip(dl_ori, dl_dst), disable=not verbose, total=ceil(chunk_n/bs)):
                            for key in _ktc:
                                data_t[key] = data_in[key]

                        shards.append(ptd)

                self._dss[ds_key] = _StackedDS(ori=_ShardedPTD(shards))

        return

    def __parse_subsample(self, **kwargs):
        '''
        Takes an already-parsed dataset and creates a subsampled version, saving the sampled
        shards at `self.path/dss.<ds_key>_sub_<pct>` (e.g. `dss.test_sub_10` for 10% of `test`).
        The new key `<ds_key>_sub_<pct>` is registered in `self._dss` and can be used as a
        loader in subsequent calls to `parse_inference()` or `load_only()`.

        Args:
        - loaders (list[str]): base ds_keys to subsample (must be present in `self._dss` or `self._dss_ori`).
        - percentage (float | dict{str: float}): fraction of samples to keep. Pass a single float to apply the same percentage to all loaders, or a dict keyed by loader name for per-loader percentages (e.g. `{'val': 0.1, 'test': 0.2}`).
        - seed (int | None): random seed for reproducibility. Defaults to None.
        - chunk_size (int | None): if set, split output into shards of at most `chunk_size` samples. Defaults to None (single shard).
        - batch_size (int): batch size used when copying data. Defaults to 64.
        - n_threads (int): `num_workers` passed to DataLoader. Defaults to 1.
        - verbose (bool): print progress messages. Defaults to False.
        '''
        self.check_uncontexted()

        loaders = kwargs.get('loaders')
        percentage = kwargs.get('percentage')
        seed = kwargs.get('seed', None)
        chunk_size = kwargs.get('chunk_size', None)
        bs = kwargs.get('batch_size', 64)
        n_threads = kwargs.get('n_threads', 1)
        verbose = kwargs.get('verbose', False)

        if isinstance(percentage, float) or isinstance(percentage, int):
            percentage = {k: percentage for k in loaders}

        if seed is not None:
            torch.manual_seed(seed)

        for ds_key in loaders:
            if ds_key not in percentage:
                raise RuntimeError(f'No percentage specified for loader "{ds_key}".')

            if ds_key in self._dss:
                src = self._dss[ds_key].ori
            elif ds_key in self._dss_ori:
                src = self._dss_ori[ds_key].ori
            else:
                raise RuntimeError(f'Dataset key "{ds_key}" not found. Call load_only() or parse_dataset() first.')

            n_total = len(src)
            perc = percentage[ds_key]
            n_sub = max(1, int(n_total * perc))
            pct = int(perc * 100)
            sub_key = f'{ds_key}_sub_{pct}'
            sub_folder = self.path / f'dss.{sub_key}'

            if verbose: print(f'\n ---- Subsampling {ds_key}: {n_sub}/{n_total} samples ({pct}%)\n')

            _chunk_size = chunk_size if chunk_size is not None else n_sub
            n_chunks = ceil(n_sub / _chunk_size)

            sample = src[0:1]
            keys = list(sample.keys())

            shards = []
            if sub_folder.exists():
                existing_chunks = sorted(sub_folder.glob('chunk_*'), key=lambda p: int(p.name.split('_')[-1]))
                if verbose: print(f'All {len(existing_chunks)} chunks for {sub_key} already exist. Loading from disk.')
                for chunk_path in existing_chunks:
                    ptd = PersistentTensorDict.from_h5(chunk_path, mode='r+')
                    ptd.batch_size = torch.Size((len(ptd),))
                    shards.append(ptd)
            else:
                indices = torch.randperm(n_total)[:n_sub].tolist()
                sub_folder.mkdir(parents=True, exist_ok=True)

                for chunk_i in range(n_chunks):
                    chunk_start = chunk_i * _chunk_size
                    chunk_end = min(chunk_start + _chunk_size, n_sub)
                    chunk_n = chunk_end - chunk_start
                    chunk_indices = indices[chunk_start:chunk_end]
                    chunk_path = sub_folder / f'chunk_{chunk_i}'

                    if verbose: print(f'  Chunk {chunk_i}: Creating ({chunk_n} samples).')
                    ptd = PersistentTensorDict(filename=chunk_path, batch_size=[chunk_n], mode='w')

                    for key in keys:
                        ptd[key] = MMT.empty(
                            shape=torch.Size((chunk_n,) + sample[key].shape[1:]),
                            dtype=sample[key].dtype
                        )

                    ptd.close()
                    ptd = PersistentTensorDict.from_h5(chunk_path, mode='r+')

                    dl_ori = DataLoader(
                        dataset=Subset(src, chunk_indices),
                        batch_size=bs,
                        collate_fn=lambda x: x,
                        shuffle=False,
                        num_workers=n_threads
                    )
                    dl_dst = DataLoader(
                        dataset=ptd,
                        batch_size=bs,
                        collate_fn=lambda x: x,
                        shuffle=False,
                        num_workers=n_threads
                    )

                    if verbose: print(f'Copying chunk {chunk_i}')
                    for data_in, data_t in tqdm(zip(dl_ori, dl_dst), disable=not verbose, total=ceil(chunk_n/bs)):
                        for key in keys:
                            data_t[key] = data_in[key]

                    shards.append(ptd)

            self._dss[sub_key] = _StackedDS(ori=_ShardedPTD(shards))

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
        self.check_uncontexted()
        
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

        self.__close()
        self._dss = {}
        self._dss_ori = {}

        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting data from {ds_key}\n')

            # detect sharded vs single-file layout
            chunk_paths = sorted((self.path / f'dss.{ds_key}').glob('chunk_*'), key=lambda p: int(p.name.split('_')[-1]))
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
                self._dss[ds_key].set_transform(transforms[ds_key] if transforms != None and ds_key in transforms else None)
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
                    if verbose: print(f'Loading inference for {inf_ds_key} from {inf_path}.')
                    _td = PersistentTensorDict.from_h5(inf_path, mode=mode)
                    _td.batch_size = torch.Size((len(_td),))

                    self._dss[inf_ds_key].stack_inference(inf=_td)

                _n_samples = len(self._dss[inf_ds_key])
            if verbose: print('loaded n_samples: ', _n_samples)
        return
    
    def __close(self):
        for ds_key in self._dss_ori:
            self._dss_ori[ds_key].close()

        for ds_key in self._dss:
            self._dss[ds_key].close()
        return

    def __enter__(self):
        self._is_contexted = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__close()
        self._is_contexted = False 
        return

    def check_uncontexted(self):
        if not self._is_contexted:
            raise RuntimeError('Function should be called within context manager')
        return
