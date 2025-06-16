# python stuff
from pathlib import Path
from tqdm import tqdm
from math import ceil

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

from peepholelib.peepholes import drill_base as driller
from collections import defaultdict


class Peepholes:
    def __init__(self, **kwargs):
        self.path = Path(kwargs['path'])
        self.name = kwargs['name']
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        # Set in get_peepholes() 
        self.target_modules = None # list of peep modules
        self._drillers = None 

        # computed in get_peepholes
        self._phs = {} 
        
        # computed in get_dataloaders()
        self._loaders = None

        # Set on __enter__() and __exit__()
        # read before each function
        self._is_contexted = False

        return

    def get_peepholes(self, **kwargs):
        '''
        Compute peepholes given `corevectors` and `drillers`.
        
        Args:
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): Parsed datasets respective the `coreVectors`.
        - corevectors (peepholelib.coreVectors.coreVectors.coreVectors): corevectors respective the `datasets`.
        - loaders (list[str]): list of loaders, usually `['train', 'val', 'test']`. If `None` uses all loaders in `corevectors._corevds.keys()`. Defaults to dss `None`.
        - target_modules (list[str]): list of modules to consider as in `model.state_dict`.
        - drillers (dict(str: peepholelib.peepholes.drill_base.DrillBase)):Dictionary where keys are the modules as in `model.state_dict` and values are classes extending `DrillBase`.
        - batchsize (int): batchsize to process `corevectors` into `peepholes`. Defaults to 64.
        - n_threads (int): Number of threads to pass as `num_workers` to `torch.utils.data.DataLoader`. Defaults to 1.
        - verbose (bool): print progress messages
        '''
        self.check_uncontexted()
        
        datasets = kwargs['datasets']
        corevectors = kwargs['corevectors']
        loaders  = kwargs.get('loaders', None)
        self.target_modules = kwargs['target_modules'] # list of peep modules
        self._drillers = kwargs['drillers']

        bs = kwargs.get('batch_size', 64)
        n_threads = kwargs.get('n_threads', 1)
        verbose = kwargs.get('verbose', False)

        if loaders == None: loaders = list(corevectors._corevds.keys())

        for ds_key in loaders:
            cvds = corevectors._corevds[ds_key]
            dssds = datasets._dss[ds_key]

            if verbose: print(f'\n ---- Getting peepholes for {ds_key}\n')
            file_path = self.path/(self.name+'.'+ds_key)
            
            # create/load PersistentTensorDict file
            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
                n_samples = len(self._phs[ds_key])
                if verbose: print('loaded n_samples: ', n_samples)
            else:
                n_samples = len(cvds)
                if verbose: print('loader n_samples: ', n_samples) 
                self.path.mkdir(parents=True, exist_ok=True)
                self._phs[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
            
            modules_to_compute = []
            for module in self.target_modules:
                if not module in self._phs[ds_key]:
                    #------------------------
                    # Pre-allocate peepholes
                    #------------------------
                    if verbose: print('allocating peepholes for module: ', module)
                    self._phs[ds_key][module] = TensorDict(batch_size=n_samples)
                    self._phs[ds_key][module]['peepholes'] = MMT.empty(shape=(n_samples, self._drillers[module].nl_model))
                    modules_to_compute.append(module)
                else:
                    if verbose: print(f'Peepholes for {module} already present. Skipping.')

            # Close PTD create with mode 'w' and re-open it with mode 'r+'
            # This is done so we can use multiple workers for reading and writting
            self._phs[ds_key].close()
            self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            #------------------------ 
            # computing peepholes
            #------------------------
            # create dataloaders
            dl_phs = DataLoader(self._phs[ds_key], batch_size=bs, collate_fn=lambda x:x, num_workers = n_threads)
            dl_cvs = DataLoader(cvds, batch_size=bs, collate_fn=lambda x: x, num_workers = n_threads)
            dl_dss = DataLoader(dssds, batch_size=bs, collate_fn=lambda x: x, num_workers = n_threads)

            if len(modules_to_compute) == 0:
                if verbose: print(f'No modules to compute for {ds_key}. Skipping.')
                continue

            if verbose: print(f'\n ---- computing peepholes for modules {modules_to_compute}\n')
            for _cvs, _dss, phs in tqdm(zip(dl_cvs, dl_dss, dl_phs), disable=not verbose, total=ceil(n_samples/bs)):
                for module in modules_to_compute:
                    phs[module]['peepholes'] = self._drillers[module](cvs=_cvs, dss=_dss)

        return 
    
    def get_feature_peepholes(self, **kwargs):
        '''
        Compute peepholes given `corevectors` and `drillers`.

        Behavior:
        - User passes `target_modules` as LAYERS (e.g. ["features.0", "classifier.3"]).
        - We expand each layer into per-window module keys: "layer.w0", "layer.w1", ...
        - Each expanded module key stores peepholes with shape [N, nl_model] (2D).
        - File names include ".windows" suffix to avoid collisions with older (layer-mode) files.

        Required kwargs:
        - datasets
        - corevectors
        - target_modules (list of layer names)
        - drillers (dict layer -> driller instance)

        Optional kwargs:
        - loaders (list[str]) default: all loaders in corevectors
        - batch_size (int)
        - n_threads (int)
        - verbose (bool)
        '''
        self.check_uncontexted()

        datasets = kwargs['datasets']
        corevectors = kwargs['corevectors']
        loaders = kwargs.get('loaders', None)

        # User passes layers here
        target_layers = kwargs['target_modules']
        self._drillers = kwargs['drillers']

        bs = kwargs.get('batch_size', 64)
        n_threads = kwargs.get('n_threads', 1)
        verbose = kwargs.get('verbose', False)

        if loaders is None:
            loaders = list(corevectors._corevds.keys())

        # -----------------------------
        # Expand layers -> window-modules
        # -----------------------------
        expanded_modules = []
        module_to_layer_window = {}

        for layer in target_layers:
            if layer not in self._drillers:
                raise KeyError(f'Layer "{layer}" not found in drillers dict.')

            dr = self._drillers[layer]

            if (not hasattr(dr, '_classifiers')) or (len(dr._classifiers) == 0):
                raise RuntimeError(
                    f'Driller for layer "{layer}" has no window classifiers. '
                    f'Fit/load it before calling get_peepholes.'
                )

            W = len(dr._classifiers)
            for wi in range(W):
                m = f'{layer}.w{wi}'
                expanded_modules.append(m)
                module_to_layer_window[m] = (layer, wi)

        # Now modules are windows
        self.target_modules = expanded_modules
        self._module_to_layer_window = module_to_layer_window

        for ds_key in loaders:
            cvds = corevectors._corevds[ds_key]
            dssds = datasets._dss[ds_key]

            if verbose:
                print(f'\n ---- Getting peepholes (WINDOW MODE) for {ds_key}\n')

            # Use different filename to avoid collisions with old (layer-mode) PTDs
            file_path = self.path / (self.name + '.' + ds_key + '.windows')

            # create/load PersistentTensorDict file
            if file_path.exists():
                if verbose:
                    print(f'File {file_path} exists. Loading from disk.')
                self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
                n_samples = len(self._phs[ds_key])
                if verbose:
                    print('loaded n_samples: ', n_samples)
            else:
                n_samples = len(cvds)
                if verbose:
                    print('loader n_samples: ', n_samples)
                self.path.mkdir(parents=True, exist_ok=True)
                self._phs[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')

            modules_to_compute = []
            for module in self.target_modules:
                if module not in self._phs[ds_key]:
                    # ------------------------
                    # Pre-allocate peepholes
                    # ------------------------
                    layer, wi = self._module_to_layer_window[module]
                    C = int(self._drillers[layer].nl_model)

                    if verbose:
                        print('allocating peepholes for module: ', module, f'(layer={layer}, window={wi})')

                    self._phs[ds_key][module] = TensorDict(batch_size=n_samples)
                    self._phs[ds_key][module]['peepholes'] = MMT.empty(shape=(n_samples, C))
                    modules_to_compute.append(module)
                else:
                    if verbose:
                        print(f'Peepholes for {module} already present. Skipping.')

            # Close PTD created with mode 'w' and re-open it with mode 'r+'
            self._phs[ds_key].close()
            self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            # ------------------------
            # computing peepholes
            # ------------------------
            dl_phs = DataLoader(
                self._phs[ds_key],
                batch_size=bs,
                collate_fn=lambda x: x,
                num_workers=n_threads
            )
            dl_cvs = DataLoader(
                cvds,
                batch_size=bs,
                collate_fn=lambda x: x,
                num_workers=n_threads
            )
            dl_dss = DataLoader(
                dssds,
                batch_size=bs,
                collate_fn=lambda x: x,
                num_workers=n_threads
            )

            if len(modules_to_compute) == 0:
                if verbose:
                    print(f'No modules to compute for {ds_key}. Skipping.')
                continue

            if verbose:
                print(f'\n ---- computing peepholes for window-modules {modules_to_compute}\n')

            total_batches = int(ceil(n_samples / bs))

            # Build iterator and wrap with tqdm only if verbose.
            iterator = zip(dl_cvs, dl_dss, dl_phs)

            if verbose:
                with tqdm(iterator, total=total_batches) as t:
                    for _cvs, _dss, phs in t:
                        for module in modules_to_compute:
                            layer, wi = self._module_to_layer_window[module]
                            # driller should accept window_idx and return [B, C]
                            phs[module]['peepholes'] = self._drillers[layer](cvs=_cvs, dss=_dss, window_idx=wi)
            else:
                for _cvs, _dss, phs in iterator:
                    for module in modules_to_compute:
                        layer, wi = self._module_to_layer_window[module]
                        phs[module]['peepholes'] = self._drillers[layer](cvs=_cvs, dss=_dss, window_idx=wi)

        return


    def get_feature_peepholes_2(self, **kwargs):
        """
        Quick-fixed get_peepholes: when a driller returns a 3D tensor [W, B, C],
        compute it once per layer and split it into per-window keys:
        layer.w0, layer.w1, ..., each stores [N, C] as before.

        Required kwargs:
        - datasets
        - corevectors
        - target_modules (list of layer names)
        - drillers (dict layer -> driller instance)

        Optional kwargs:
        - loaders, batch_size, n_threads, verbose
        """
        self.check_uncontexted()

        datasets = kwargs['datasets']
        corevectors = kwargs['corevectors']
        loaders = kwargs.get('loaders', None)

        target_layers = kwargs['target_modules']
        self._drillers = kwargs['drillers']

        bs = kwargs.get('batch_size', 64)
        n_threads = kwargs.get('n_threads', 1)
        verbose = kwargs.get('verbose', False)

        if loaders is None:
            loaders = list(corevectors._corevds.keys())

        # Expand layers -> window-modules as before
        expanded_modules = []
        module_to_layer_window = {}

        for layer in target_layers:
            if layer not in self._drillers:
                raise KeyError(f'Layer "{layer}" not found in drillers dict.')

            dr = self._drillers[layer]
            if (not hasattr(dr, '_classifiers')) or (len(dr._classifiers) == 0):
                raise RuntimeError(
                    f'Driller for layer "{layer}" has no window classifiers. Fit/load it first.'
                )

            W = len(dr._classifiers)
            for wi in range(W):
                m = f'{layer}.w{wi}'
                expanded_modules.append(m)
                module_to_layer_window[m] = (layer, wi)

        self.target_modules = expanded_modules
        self._module_to_layer_window = module_to_layer_window

        for ds_key in loaders:
            cvds = corevectors._corevds[ds_key]
            dssds = datasets._dss[ds_key]

            if verbose:
                print(f'\n ---- Getting peepholes (WINDOW MODE) for {ds_key}\n')

            file_path = self.path / (self.name + '.' + ds_key + '.windows')

            if file_path.exists():
                if verbose:
                    print(f'File {file_path} exists. Loading from disk.')
                self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
                n_samples = len(self._phs[ds_key])
                if verbose:
                    print('loaded n_samples: ', n_samples)
            else:
                n_samples = len(cvds)
                if verbose:
                    print('loader n_samples: ', n_samples)
                self.path.mkdir(parents=True, exist_ok=True)
                self._phs[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')

            modules_to_compute = []
            for module in self.target_modules:
                if module not in self._phs[ds_key]:
                    layer, wi = self._module_to_layer_window[module]
                    C = int(self._drillers[layer].nl_model)
                    if verbose:
                        print('allocating peepholes for module: ', module, f'(layer={layer}, window={wi})')
                    self._phs[ds_key][module] = TensorDict(batch_size=n_samples)
                    self._phs[ds_key][module]['peepholes'] = MMT.empty(shape=(n_samples, C))
                    modules_to_compute.append(module)
                else:
                    if verbose:
                        print(f'Peepholes for {module} already present. Skipping.')

            # Reopen for r+ after possibly creating new file
            self._phs[ds_key].close()
            self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            if len(modules_to_compute) == 0:
                if verbose:
                    print(f'No modules to compute for {ds_key}. Skipping.')
                continue

            if verbose:
                print(f'\n ---- computing peepholes for window-modules {modules_to_compute}\n')

            # Group modules by layer so we compute each layer once per batch
            layer_to_modules = defaultdict(list)
            for module in modules_to_compute:
                layer, wi = self._module_to_layer_window[module]
                layer_to_modules[layer].append((module, wi))

            dl_phs = DataLoader(self._phs[ds_key], batch_size=bs, collate_fn=lambda x: x, num_workers=n_threads)
            dl_cvs = DataLoader(cvds, batch_size=bs, collate_fn=lambda x: x, num_workers=n_threads)
            dl_dss = DataLoader(dssds, batch_size=bs, collate_fn=lambda x: x, num_workers=n_threads)

            total_batches = int(ceil(n_samples / bs))
            iterator = zip(dl_cvs, dl_dss, dl_phs)

            if verbose:
                iterator = tqdm(iterator, total=total_batches)

            # Main loop: for each batch, compute per layer once, then split to windows
            for _cvs, _dss, phs in iterator:
                # for each layer that needs computing in this dataset, call driller once
                for layer, module_wi_list in layer_to_modules.items():
                    # call driller once — do NOT pass window_idx so we get full [W, B, C] output
                    out = self._drillers[layer](cvs=_cvs, dss=_dss)

                    # make sure out is a torch Tensor
                    if not torch.is_tensor(out):
                        raise RuntimeError(f"Driller for layer {layer} returned non-tensor type: {type(out)}")

                    # Expected shapes:
                    # * preferred: [W, B, C]  -> split on dim 0
                    # * alternate: [B, C]     -> single-window case (W==1)
                    if out.dim() == 3:
                        W, B, C = out.shape
                        # assign each requested window slice
                        for (module, wi) in module_wi_list:
                            if not (0 <= wi < W):
                                raise IndexError(f"Requested wi={wi} but driller returned W={W}")
                            slice_bc = out[wi]  # shape [B, C]
                            # ensure CPU (PersistentTensorDict likely stores CPU)
                            slice_bc_cpu = slice_bc.detach().cpu()
                            phs[module]['peepholes'] = slice_bc_cpu
                    elif out.dim() == 2:
                        # single-window driller output [B, C] — assign to all windows requested
                        B, C = out.shape
                        for (module, wi) in module_wi_list:
                            slice_bc_cpu = out.detach().cpu()
                            phs[module]['peepholes'] = slice_bc_cpu
                    else:
                        raise RuntimeError(f"Unexpected driller output shape {out.shape} for layer {layer}")

        return


    def load_only(self, **kwargs):
        '''
        Load the peepholes (robust to both layer-based and window-based formats).
        '''
        self.check_uncontexted()

        verbose = kwargs.get('verbose', False)
        loaders = kwargs['loaders']
        mode = kwargs.get('mode', 'r')

        for ds_key in loaders:
            if verbose:
                print(f'\n ---- Loading peepholes for {ds_key}\n')

            # Prefer window-based peepholes if present
            file_path_windows = self.path / (self.name + '.' + ds_key + '.windows')
            file_path_legacy = self.path / (self.name + '.' + ds_key)

            if file_path_windows.exists():
                file_path = file_path_windows
                if verbose:
                    print(f'Loading WINDOW-based peepholes from {file_path}')
            elif file_path_legacy.exists():
                file_path = file_path_legacy
                if verbose:
                    print(f'Loading LEGACY (layer-based) peepholes from {file_path}')
            else:
                raise FileNotFoundError(
                    f'No peephole file found for loader "{ds_key}". '
                    f'Tried:\n  - {file_path_windows}\n  - {file_path_legacy}'
                )

            self._phs[ds_key] = PersistentTensorDict.from_h5(file_path, mode=mode)

        return

    def get_conceptograms(self, **kwargs):
        '''
        Get conceptograms from peepholes. A conceptogram is the concatenation of peepholes for multiple modules.
        
        Args:
        - target_modules (list[str]): list of target module keys
        - loaders (list[str]): list of loaders (usually 'train', 'test', 'val' within self._phs 
        - verbose (bool): print progress information
        '''
        self.check_uncontexted()
        
        target_modules = kwargs.get('target_modules', None)
        verbose = kwargs.get('verbose', False)

        if self._phs == None:
            raise RuntimeError('Peepholes not present. Please run get_peepholes() first.')

        loaders = kwargs.get('loaders', list(self._phs))

        _conceptograms = {}
        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting conceptograms for {ds_key}\n')
            file_path = self.path / (self.name + '.' + ds_key)

            #-----------------------------------------
            # Check if peepholes exist before computing scores
            #-----------------------------------------
            n_samples = len(self._phs[ds_key])

            if target_modules == None:
                target_modules = self._phs[ds_key].keys()

            for module in target_modules:
                if module not in self._phs[ds_key]:
                    raise ValueError(f"Peepholes for module {module} do not exist. Please run get_peepholes() first.")

                if 'peepholes' not in self._phs[ds_key][module]:
                    raise ValueError(f"Peepholes do not exist in module {module}. Please run get_peepholes() first.")

            _conceptograms[ds_key] = torch.stack([self._phs[ds_key][layer]['peepholes'] for layer in target_modules], dim=1)

        return _conceptograms

    def __enter__(self):
        self._is_contexted = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        verbose = True 

        if self._phs == None:
            if verbose: print('no peepholes to close. doing nothing.')
            return

        for ds_key in self._phs:
            if verbose: print(f'closing {ds_key}')
            self._phs[ds_key].close()
            
        self._is_contexted = False 
        return

    def check_uncontexted(self):
        if not self._is_contexted:
            raise RuntimeError('Function should be called within context manager')
        return
