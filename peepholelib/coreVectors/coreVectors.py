# torch stuff
from math import ceil
import numpy as np
import torch
from torch.utils.data import DataLoader 
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.nn.functional import mse_loss 

# generic python stuff
from pathlib import Path
from tqdm import tqdm

class CoreVectors():
     
    from .get_coreVectors import get_coreVectors

    def __init__(self, **kwargs):
        '''
        Create instance of 'corevectors'.

        Args:
        - path (str|pathlib.Path): Path to save corevectors.
        - name (str): Name for corevectors. See 'corevector.get_corevectors()' for details.
        '''
        self.path = Path(kwargs['path'])
        self.name = kwargs['name']
        
        # create folder
        self.path.mkdir(parents=True, exist_ok=True)

        self._model = kwargs['model'] if 'model' in kwargs else None  

        # computed in get_coreVectors()
        self._corevds = None 


        # set in normalize_corevectors() 
        self._norm_mean = None 
        self._norm_std = None 
        
        # Set on __enter__() and __exit__()
        # read before each function
        self._is_contexted = False

        # computed in get_dataloaders()
        self._loaders = {}
        return
     
    def normalize_corevectors(self, **kwargs):
        '''
        Normalize corevectors.

        Args:
        - batch_size (int): Creates dataloader to do computation in batch size. Defaults to 64.
        - n_threads (int): 'num_workers' passed to 'torch.utils.data.DataLoader'. Defaults to 1.
        - wrt (str): selects which loader to compute the means and stds, the other loaders are normalized using this loader's means and stds. Defaults to None. 
        - from_file (str|pathlib.Path): If 'wrt'=None, use the means and stds from this file. Defaults to None.
        - to_file (str|pathlib.Path): Save means and stds to this file. Defaults to None.
        - target_layers (list[str]): Normalize only the specified layers.
        - loaders (list[str]): Normalize only the specified loaders. If None, normalize all loaders in the corevectors.
        - verbose (bool): print progress messages.
        '''
        self.check_uncontexted()

        bs = kwargs.get('batch_size', 64) 
        n_threads = kwargs.get('n_threads', 1) 

        wrt = kwargs.get('wrt', None)
        from_file = kwargs.get('from_file', None)
        if from_file != None: Path(from_file)
        to_file = kwargs.get('to_file', None)
        if to_file != None: Path(to_file)

        target_layers = kwargs.get('target_layers', None)
        loaders = kwargs.get('loaders', None)

        verbose = kwargs.get('verbose', False) 

        if self._corevds == None:
            raise RuntimeError('No corevectors to normalize. Run get_corevectors() first.')

        if wrt == None and from_file == None:
            raise RuntimeError(f'Specify `wrt` or `from_file`.')
        
        if from_file != None:
            if verbose: print(f'Loading normalization from {from_file}')
            means, stds = torch.load(from_file, weights_only=False)
        else: # wrt will not be None
            if verbose: print(f'Computing normalization from {wrt}')
            means = self._corevds[wrt].mean(dim=0)
            
            stds = self._corevds[wrt].std(dim=0)

        if target_layers != None:
            keys_to_pop = tuple(means.keys()-target_layers)
            for k in keys_to_pop:
                means.pop(k, default=None)
                stds.pop(k, default=None)
        
        if loaders == None: loaders = self._corevds
        
        for ds_key in loaders:
            if verbose: print(f'\n ---- Normalizing core vectors for {ds_key}\n')
            dl = DataLoader(self._corevds[ds_key], batch_size=bs, collate_fn=lambda x: x, num_workers = n_threads)
            
            for batch in tqdm(dl, disable=not verbose, total=len(dl)):
                for _k in means.keys():
                    batch[_k] = (batch[_k]- means[_k])/stds[_k]

        if to_file != None:
            to_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save((means, stds), to_file)

        self._norm_mean = means
        self._norm_std = stds

        return
    
    def load_only(self, **kwargs):
        '''
        Load already computed corevectors.

        Args:
        - loadrs (list[str]): load the specified loaders
        - mode (str): Opens the file with the specified mode. See 'tensordict.PersistentTensorDict.from_h5()' for details. Defaults to 'r'.
        - norm_file (str): load the normalization information. Defaults to None. 
        - verbose (bool): print progress messages.
        '''
        self.check_uncontexted()

        loaders = kwargs['loaders']
        mode = kwargs.get('mode', 'r')
        norm_file = kwargs.get('norm_file', None)
        if norm_file != None: norm_file = Path(norm_file)
        verbose = kwargs.get('verbose', False)

        self._corevds = {}
        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting data from {ds_key}\n')
            
            _cvs_file_paths = self.path/(self.name+'.'+ds_key)

            if verbose: print(f'Loading files {_cvs_file_paths} from disk. ')
            self._corevds[ds_key] = PersistentTensorDict.from_h5(_cvs_file_paths, mode=mode)

            _n_samples = len(self._corevds[ds_key])
            if verbose: print('loaded n_samples: ', _n_samples)
       
        if norm_file != None:
            if verbose: print('Loading normalization info.')
            self._norm_mean, self._norm_std = torch.load(norm_file)

        return
    
    def __enter__(self):
        self._is_contexted = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        verbose = True 

        if self._corevds == None:
            if verbose: print('no corevds to close.')
        else:
            for ds_key in self._corevds:
                if verbose: print(f'closing {ds_key}')
                self._corevds[ds_key].close()

        self._is_contexted = False 
        return

    def check_uncontexted(self):
        if not self._is_contexted:
            raise RuntimeError('Function should be called within context manager')
        return


    def get_corruptions_all(self, **kwargs):
        '''
        Generate a corrupted version of the initial dataset using wombats package
        '''
        self.check_uncontexted()

        model = kwargs['model']
        corruptions = kwargs['corruptions']
        verbose = kwargs.get('verbose', False)
        loaders = kwargs['loaders']
        bs = kwargs.get('bs', 2**11) 
        n_threads = kwargs.get('n_threads', 8)
        n_samples_ = kwargs['n_samples']
        suffix = kwargs.get('suffix', None)
        thr = kwargs['thr']
        seed = kwargs.get('seed', 42)
        
        g = torch.Generator(device='cpu').manual_seed(seed)
        for loader in loaders:
            layers = list(self._corevds[loader].values())


            # Stack along sensor dimension
            # Result: (N, num_layers, 10)
            stacked = torch.stack(layers, dim=1)
            n_samples = len(self._corevds[loader])
            if verbose: print(f' Got {n_samples} samples from {loader}')
            data_shape = self._corevds[loader].shape
            
                
            n_channels = data_shape[1]
            #print(f'n_channels{n_channels}')
            n_corr = len(corruptions)

            '''ds_sample = self._corevds[loader][_layer]
            print(f'ds_sample{ds_sample}')
            model_input = ds_sample[0]'''
            #model_input =  model_input.view(-1, 1, 5, 2)
            model_input = stacked[0]           # (num_layers=8, 10)
            model_input = model_input.unsqueeze(0).unsqueeze(0)
            pad = torch.zeros((1,1,2,10))  # 2 missing sensors
            model_input_padded = torch.cat([model_input, pad], dim=2)  # now shape (1,1,10,10
            #print(f'model_input_padded.shape{model_input_padded.shape}')
            #quit()
            with torch.no_grad():
                _out, _ls = model(model_input_padded.to(model.device)) 
            os = _out.shape[1:]
            ls = _ls.shape[1:]

            cdsk = loader+'-cv-c-all'
            idx = torch.randperm(n_samples , generator=g)[:n_samples_]

            if suffix != None:
                cdsk += '-'+suffix

            file_path = self.path/('dss.'+cdsk) 

            if file_path.exists():
                print(f'{file_path} exists. I am not overwritting it. Skipping')
                continue

            self._corevds[cdsk] = PersistentTensorDict(filename=file_path, batch_size=[n_samples_*n_corr], mode = 'w')
            for _layer, tensor in self._corevds[loader].items():

                self._corevds[cdsk][_layer] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)+data_shape), dtype=torch.float32)
            self._corevds[cdsk]['data'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)+data_shape), dtype=torch.float32)
            self._corevds[cdsk]['output'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)+os), dtype=torch.float32)
            self._corevds[cdsk]['residual'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)+os), dtype=torch.float32)
            self._corevds[cdsk]['latent_space'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)+ls), dtype=torch.float32)
            self._corevds[cdsk]['corruption'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)), dtype=torch.int)
            self._corevds[cdsk]['detection'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)), dtype=torch.int)
            for c in range(n_corr):
                self._corevds[cdsk][f'corruption{c}'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)), dtype=torch.int)

            # Close PTD create with mode 'w' and re-open it with mode 'r+'
            # This is done so we can use multiple workers with the dataloaders 
            #print(self._corevds[cdsk].keys())
            self._corevds[cdsk].close()
            self._corevds[cdsk] = PersistentTensorDict.from_h5(file_path, mode='r+')
            
            corevector_dl = DataLoader(
                    dataset = self._corevds[cdsk],
                    batch_size = n_samples_,
                    collate_fn = lambda x:x,
                    shuffle = False,
                    num_workers = n_threads
                    )
            
            for _layer, tensor in self._corevds[loader].items():
                  
                for i, (corevector, (corruption, corrupter)) in enumerate(zip(corevector_dl, corruptions.items())):
                    t = self._corevds[loader][_layer][idx].clone()

                    for channel in range(n_channels):
                        sig = self._corevds[loader][_layer][idx].detach().cpu().numpy()
                        #print(f'sig.shape{sig.shape}')
                        #quit()
                        #print("sig type:", type(sig))
                        #print("sig shape:", sig.shape)
                        #print("sig ndim:", sig.ndim)
                        corrupter.fit(sig)
                        #print("Q shape:", corrupter.Q.shape if hasattr(corrupter, "Q") else "no Q")
                        #print("Rb_theta shape:", corrupter.Rb_theta.shape if hasattr(corrupter, "Rb_theta") else "no Rb_theta")
                        #print(vars(corrupter))

                        corr_sig = corrupter.distort(sig)
                        t[:, :] = torch.from_numpy(corr_sig).to(self._corevds[loader].device)
                    #print(type(corevector))
                    #print(type(corevector['data']))
                    #print(f'corevector[data].shape{corevector['data'].shape}')
                    #print(f't.shape{t.shape}')
                    t_expanded = t.unsqueeze(1).expand(-1, 211314, -1)  # shape: [1000, 211314, 10]
                    corevector['data'] = t_expanded
                    #corevector['data'] = t
                    corevector['corruption'] = torch.ones(len(t))*i
                    corevector[f'corruption{i}'] = torch.ones(len(t))

            corevector_dl = DataLoader(
                    dataset = self._corevds[cdsk],
                    batch_size = bs,
                    collate_fn = lambda x:x,
                    shuffle = False,
                    num_workers = n_threads
                    )
        
            for corevector in tqdm(corevector_dl, disable=not verbose, total=ceil(n_samples_*n_corr/bs)):
                with torch.no_grad():
                    x = corevector[_layer].to(model.device)
                    with torch.no_grad():
                        x =  x.unsqueeze(1) #[5000, 1, 211314, 10]
                        y, ls = model(x)
                        #print(f'y.shape{y.shape}\nls.shape{ls.shape}')
                    corevector['output'] = y
                    corevector['residual'] = y - x
                    corevector['latent_space'] = ls

                    mse = mse_loss(y, x, reduction='none')  
                    mse_per_sample = mse.view(mse.size(0), -1).mean(dim=1)
                    corevector['detection'] = (mse_per_sample > thr).long()
            
        return