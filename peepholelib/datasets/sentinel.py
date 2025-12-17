# general python stuff
import pandas as pd
from pathlib import Path
from types import NoneType
from math import floor, ceil 
from tqdm import tqdm

# tensordict
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.datasets.datasetWrap import DatasetWrap
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT

# torch stuff
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import mse_loss 

class CustomDS(Dataset):
    def __init__(self, data, labels, ws=16):
        Dataset.__init__(self) 

        _data = torch.tensor(data.values, dtype=torch.float32)
        nw = floor(_data.shape[0]/ws) # num windows

        data = _data[:ws*nw]
        data = data.reshape(-1, ws, data.shape[-1]) # 16 is the number of signals
        data = data.permute(0, 2, 1).unsqueeze(dim=1) ## B, 1, nc, nw

        idx = data.isnan().any(dim=(2,3)).logical_not()#used dim instead of axis
        self.data = data[idx].unsqueeze(dim=1)

        if type(labels) == NoneType:
            self.labels = None  
        elif type(labels) == pd.DataFrame:
            _labels = torch.tensor(labels.values, dtype=torch.float32)
            labels = _labels[:ws*nw]
            labels = labels.reshape(-1, ws, labels.shape[-1])
            
            self.labels = labels[idx.squeeze(dim=1)]
        else: 
            raise RuntimeError('Labels should be None or a dataframe')

        if type(self.labels) != NoneType:
            idx = (self.labels == 1).any(axis=(1, 2)).logical_not()
            self.data = self.data[idx]
            self.labels = self.labels[idx]

        # filter samples with label == 1
        self.len = self.data.shape[0]
        
        return

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.labels == None:
            return {'data': self.data[idx]}
        else:
            return {'data': self.data[idx], 'label': self.labels[idx]}
        
class SentinelWrap(DatasetWrap):#DatasetBase

    def __init__(self, **kwargs):
        self.path = kwargs.get('path')
        return

    def __load_data__(self, **kwargs):
        ws = kwargs.get('window_size', 16) 
        seed = kwargs.get('seed', 42)
        n_samples = kwargs.get('n_samples', None)
        split = kwargs.get('split', 0.2)
        verbose = kwargs.get('verbose', False)

        # load data and the labels
        train_file = Path(self.path)/'train_data.pkl'
        test_file = Path(self.path)/'test_data.pkl'
        label_file = Path(self.path)/'test_labels.pkl'

        data_train = pd.read_pickle(train_file.as_posix())
        data_test = pd.read_pickle(test_file.as_posix())
        test_labels = pd.read_pickle(label_file.as_posix())

        torch.manual_seed(seed)
        self.__dataset__ = {}
        
        # split train into train and val
        ds_train = CustomDS(data=data_train, labels=None, ws=ws)
        self.__dataset__['train'], self.__dataset__['val'] = torch.utils.data.random_split(
                ds_train,
                [1 - split, split],
                generator = torch.Generator().manual_seed(seed)
        )

        self.__dataset__['test'] = CustomDS(data=data_test, labels=test_labels, ws=ws)
        
        if n_samples != None:
            for ds_key in n_samples.keys():
                if n_samples[ds_key] != None:
                    perc = n_samples[ds_key]/len(self.__dataset__[ds_key])
                    self.__dataset__[ds_key], _ = torch.utils.data.random_split(
                            self.__dataset__[ds_key],
                            [perc, 1-perc],
                            generator = torch.Generator().manual_seed(seed)
                            ) 

        return

    def get(self, ds_key, idx):
        '''
        Get item from the dataset.
        
        Args:
        - idx (int): Index of the item to get.
        - ds_key (str): Key of the dataset to get the item from ('train', 'val', 'test').
        
        Returns:
        - a tuple of (image, label)
        '''
        if not self._dss:
            raise RuntimeError('Data not loaded. Please run load_data() first.')
        
        return [self._dss[ds_key][idx]]
    
class Sentinel(ParsedDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
    
    @classmethod
    def create_ds(cls, **kwargs):
        path = Path(kwargs['path'])
        sentinel_wrap = kwargs['sentinel_wrap']
        bs = kwargs.get('batch_size', 2**11)
        verbose = kwargs.get('verbose', False)
        
        path.mkdir(parents=True, exist_ok=True)
        cls_inst = cls(path = path)
        cls_inst._dss = {}
        
        for ds_key in sentinel_wrap.__dataset__:
            file_path = cls_inst.path/('dss.'+ds_key)
            n_samples = len(sentinel_wrap.__dataset__[ds_key])
            
            # check if PTD exists 
            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                cls_inst._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            else:
                if verbose: print(f'Creating {ds_key} dataset with n_samples: ', n_samples)
                cls_inst._dss[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
                
                # get sample to get shapes
                sample = sentinel_wrap.__dataset__[ds_key][0]
                for key in sample.keys():
                    if verbose: print(f'allocating {key} with shape {sample[key].shape}')
                    cls_inst._dss[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,)+sample[key].shape), dtype=torch.float32)
        
                # create Dataloader of input dataset
                ds_in = DataLoader(
                    dataset = sentinel_wrap.__dataset__[ds_key],
                    batch_size = bs
                )
                                                                                                                                        
                ds_t = DataLoader(
                    cls_inst._dss[ds_key],
                    collate_fn = lambda x:x, 
                    batch_size = bs
                )
                                                                                                                                        
                for data_in, data_t in tqdm(zip(ds_in, ds_t), disable=not verbose, total=ceil(n_samples/bs)):
                    for key in data_in.keys():
                        data_t[key] = data_in[key]
            
            # close the PTD
            cls_inst._dss[ds_key].close()
        return

    # overwrite the parse_ds() from peepholelib.datasets.parsedDataset.ParsedDataset
    def parse_ds(self, **kwargs):
        self.check_uncontexted()

        model = kwargs['model']
        loaders = kwargs.get('loaders', None)
        bs = kwargs.get('batch_size', 2**11)
        verbose = kwargs.get('verbose', False)
        
        if loaders == None:
            loaders = self._dss.keys()

        for ds_key in loaders:
            file_path = self.path/('dss.'+ds_key)
            n_samples = len(self._dss[ds_key])
            
            # dataset sample for dry run
            sample = self._dss[ds_key][0:1]['data'].to(model.device)
            with torch.no_grad():
                _out, _ls = model(sample)
                     
            os = _out.shape[1:]
            ls = _ls.shape[1:]

            # check and skip if the values are already there
            if ('output' in self._dss[ds_key]) and ('residual' in self._dss[ds_key]) and ('latent_space' in self._dss[ds_key]):
                continue

            # need to fix the batch size - workaround  
            self._dss[ds_key].batch_size = torch.Size((n_samples,))
            # allocate disk space
            self._dss[ds_key]['output'] = MMT.empty(shape=torch.Size((n_samples,)+os), dtype=torch.float32)
            self._dss[ds_key]['residual'] = MMT.empty(shape=torch.Size((n_samples,)+os), dtype=torch.float32)
            self._dss[ds_key]['latent_space'] = MMT.empty(shape=torch.Size((n_samples,)+ls), dtype=torch.float32)

            dl = DataLoader(
                self._dss[ds_key],
                collate_fn = lambda x:x, 
                batch_size = bs
            )

            for data in tqdm(dl, disable=not verbose, total=ceil(n_samples/bs)):
                #compute predictions which is the out of decoder
                with torch.no_grad():
                    x = data['data'].float().to(model.device)
                    with torch.no_grad():
                        y, ls = model(x)
                    
                    data['output'] = y
                    data['residual'] = y - x 
                    data['latent_space'] = ls           
        
        return
    
    def get_corruptions_all(self, **kwargs):
        '''
        Generate a corrupted version of the initial dataset using wombats package
        '''
        self.check_uncontexted()

        model = kwargs.get('model')
        corruptions = kwargs.get('corruptions')
        verbose = kwargs.get('verbose', False)
        loaders = kwargs.get('loaders')
        bs = kwargs.get('bs', 2**11) 
        n_threads = kwargs.get('n_threads', 8)
        n_samples_ = kwargs.get('n_samples')
        suffix = kwargs.get('suffix', None)
        thr = kwargs.get('thr')
        seed = kwargs.get('seed', 42)
        
        g = torch.Generator(device='cpu').manual_seed(seed)

        for loader in loaders:
            n_samples = len(self._dss[loader])
            if verbose: print(f' Got {n_samples} samples from {loader}')

            data_shape = self._dss[loader]['data'][0].shape
            n_channels = data_shape[1]
            n_corr = len(corruptions)

            ds_sample = self._dss[loader][0:1]
            model_input = ds_sample['data']

            with torch.no_grad():
                _out, _ls = model(model_input.to(model.device)) 
            os = _out.shape[1:]
            ls = _ls.shape[1:]

            cdsk = loader+'-c-all'
            idx = torch.randperm(n_samples , generator=g)[:n_samples_]

            if suffix != None:
                cdsk += '-'+suffix

            file_path = self.path/('dss.'+cdsk) 

            if file_path.exists():
                print(f'{file_path} exists. I am not overwritting it. Skipping')
                continue

            self._dss[cdsk] = PersistentTensorDict(filename=file_path, batch_size=[n_samples_*n_corr], mode = 'w')

            self._dss[cdsk]['data'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)+data_shape), dtype=torch.float32)
            self._dss[cdsk]['output'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)+os), dtype=torch.float32)
            self._dss[cdsk]['residual'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)+os), dtype=torch.float32)
            self._dss[cdsk]['latent_space'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)+ls), dtype=torch.float32)
            self._dss[cdsk]['corruption'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)), dtype=torch.int)
            self._dss[cdsk]['detection'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)), dtype=torch.int)
            for c in range(n_corr):
                self._dss[cdsk][f'corruption{c}'] = MMT.empty(shape=torch.Size((n_samples_*n_corr,)), dtype=torch.int)

            # Close PTD create with mode 'w' and re-open it with mode 'r+'
            # This is done so we can use multiple workers with the dataloaders 
            self._dss[cdsk].close()
            self._dss[cdsk] = PersistentTensorDict.from_h5(file_path, mode='r+')
            
            sentinel_dl = DataLoader(
                    dataset = self._dss[cdsk],
                    batch_size = n_samples_,
                    collate_fn = lambda x:x,
                    shuffle = False,
                    num_workers = n_threads
                    )
            
            for i, (sentinel, (corruption, corrupter)) in enumerate(zip(sentinel_dl, corruptions.items())):
                t = self._dss[loader]['data'][idx].clone()

                for channel in range(n_channels):
                    sig = self._dss[loader]['data'][idx , 0, channel, :].detach().cpu().numpy()
                    corrupter.fit(sig)
                    corr_sig = corrupter.distort(sig)
                    t[:, 0, channel, :] = torch.from_numpy(corr_sig).to(self._dss[loader].device)

                sentinel['data'] = t
                sentinel['corruption'] = torch.ones(len(t))*i
                sentinel[f'corruption{i}'] = torch.ones(len(t))

            sentinel_dl = DataLoader(
                    dataset = self._dss[cdsk],
                    batch_size = bs,
                    collate_fn = lambda x:x,
                    shuffle = False,
                    num_workers = n_threads
                    )
        
            for sentinel in tqdm(sentinel_dl, disable=not verbose, total=ceil(n_samples_*n_corr/bs)):
                with torch.no_grad():
                    x = sentinel['data'].float().to(model.device)
                    with torch.no_grad():
                        y, ls = model(x)

                    sentinel['output'] = y
                    sentinel['residual'] = y - x
                    sentinel['latent_space'] = ls

                    mse = mse_loss(y, x, reduction='none')  
                    mse_per_sample = mse.view(mse.size(0), -1).mean(dim=1)
                    sentinel['detection'] = (mse_per_sample > thr).long()
            
        return
    
    def get_corruptions_single(self, **kwargs):
        '''
        Generate a corrupted version of the initial dataset using wombats package
        '''
        self.check_uncontexted()

        model = kwargs.get('model')
        corruptions = kwargs.get('corruptions')
        verbose = kwargs.get('verbose', False)
        loaders = kwargs.get('loaders') 
        n_threads = kwargs.get('n_threads', 8)
        bs = kwargs.get('bs', 2**11) 
        n_samples_ = kwargs.get('n_samples')
        suffix = kwargs.get('suffix', None)
        thr = kwargs.get('thr')
        seed = kwargs.get('seed', 42)

        g = torch.Generator(device='cpu').manual_seed(seed)

        for loader in loaders:
            n_samples = len(self._dss[loader])
            if verbose: print(f' Got {n_samples} samples from {loader}')

            data_shape = self._dss[loader]['data'][0].shape
            n_channels = data_shape[1]
            n_corr = len(corruptions)

            configs = []
            for corr_id, corr_key in enumerate(corruptions.values()):
                for ch in range(n_channels):
                    configs.append({
                        'channel': ch,
                        'corruption': corr_key,
                        'corr_id': corr_id
                    })

            # dry run to get shapes
            sample = self._dss[loader][0:1]['data'].to(model.device)
            with torch.no_grad():
                _out, _ls = model(sample) 
            os = _out.shape[1:]
            ls = _ls.shape[1:]
            
            idx  = torch.randperm(n_samples , generator=g)[:n_samples_]
            cdsk = loader+'-c-single'

            if suffix != None:
                cdsk += '-'+suffix

            file_path = self.path/('dss.'+cdsk) 

            if file_path.exists():
                print(f'{file_path} exists. I am not overwritting it. Skipping')
                continue

            self._dss[cdsk] = PersistentTensorDict(filename=file_path, batch_size=[n_samples_*n_corr*n_channels], mode = 'w')

            self._dss[cdsk]['data'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_channels,)+data_shape), dtype=torch.float32)
            self._dss[cdsk]['output'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_channels,)+os), dtype=torch.float32)
            self._dss[cdsk]['residual'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_channels,)+os), dtype=torch.float32)
            self._dss[cdsk]['latent_space'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_channels,)+ls), dtype=torch.float32)
            self._dss[cdsk]['corruption'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_channels,)), dtype=torch.int)
            self._dss[cdsk]['detection'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_channels,)), dtype=torch.int)
            self._dss[cdsk]['RW'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_channels,)), dtype=torch.int)
            self._dss[cdsk]['channel'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_channels,)), dtype=torch.int)

            # Close PTD create with mode 'w' and re-open it with mode 'r+'
            # This is done so we can use multiple workers with the dataloaders 
            self._dss[cdsk].close()
            self._dss[cdsk] = PersistentTensorDict.from_h5(file_path, mode='r+')

            sentinel_dl = DataLoader(
                            dataset = self._dss[cdsk],
                            batch_size = n_samples_,
                            collate_fn = lambda x:x,
                            shuffle = False,
                            num_workers = n_threads
                        )
            
            for sentinel, config in zip(sentinel_dl, configs):
                ch = config['channel']
                corr = config['corruption']
                corr_id = config['corr_id']
                t = self._dss[loader]['data'][idx].clone()
                sig = self._dss[loader]['data'][idx, 0, ch, :].detach().cpu().numpy()

                corr.fit(sig)
                corr_sig = corr.distort(sig) 

                t[:, 0, ch, :] = torch.from_numpy(corr_sig).to(self._dss[loader].device)

                sentinel['data'] = t
                sentinel['corruption'] = torch.ones(len(t))*corr_id
                sentinel['channel'] = torch.ones(len(t))*ch
                sentinel['RW'] = torch.ones(len(t))*(ch//4)

            sentinel_dl = DataLoader(
                    dataset = self._dss[cdsk],
                    batch_size = bs,
                    collate_fn = lambda x:x,
                    shuffle = False,
                    num_workers = n_threads
                )
        
            for sentinel in tqdm(sentinel_dl, disable=not verbose, total=ceil(n_samples_*n_corr*n_channels/bs)):
                with torch.no_grad():
                    x = sentinel['data'].float().to(model.device)
                    with torch.no_grad():
                        y, ls = model(x)

                    sentinel['output'] = y
                    sentinel['residual'] = y - x
                    sentinel['latent_space'] = ls

                    mse = mse_loss(y, x, reduction='none')  
                    mse_per_sample = mse.view(mse.size(0), -1).mean(dim=1)
                    sentinel['detection'] = (mse_per_sample > thr).long()

        return
            
    def get_corruptions_RW(self, **kwargs):
        '''
        Generate a corrupted version of the initial dataset using wombats package
        '''
        self.check_uncontexted()

        model = kwargs.get('model')
        corruptions = kwargs.get('corruptions')
        verbose = kwargs.get('verbose', False)
        loaders = kwargs.get('loaders') 
        n_threads = kwargs.get('n_threads', 8)
        bs = kwargs.get('bs', 2**11) 
        n_samples_ = kwargs.get('n_samples')
        suffix = kwargs.get('suffix', None)
        thr = kwargs.get('thr')
        seed = kwargs.get('seed', 42)
        
        g = torch.Generator(device='cpu').manual_seed(seed)

        for loader in loaders:
            n_samples = len(self._dss[loader])
            if verbose: print(f' Got {n_samples} samples from {loader}')

            data_shape = self._dss[loader]['data'][0].shape

            n_channels = data_shape[1]
            n_corr = len(corruptions)
            n_rw = 4
            groups = [
                (i, list(range(i * n_rw, (i + 1) * n_rw)))
                for i in range(n_channels // n_rw)
            ]

            configs = [
                    {
                        'group_id': g_id,
                        'channels': ch_group,
                        'corruption': corr,
                        'corr_id': corr_id
                    }
                    for corr_id, corr in enumerate(corruptions.values())
                    for g_id, ch_group in groups
                ]
            
            # dry run to get shapes
            sample = self._dss[loader][0:1]['data'].to(model.device)
            with torch.no_grad():
                _out, _ls = model(sample) 
            os = _out.shape[1:]
            ls = _ls.shape[1:]

            idx  = torch.randperm(n_samples, generator=g)[:n_samples_]
            cdsk = loader+'-c-RW' 
            if suffix != None:
                cdsk += '-'+suffix

            file_path = self.path/('dss.'+cdsk) 

            if file_path.exists():
                print(f'{file_path} exists. I am not overwritting it. Skipping')
                continue

            self._dss[cdsk] = PersistentTensorDict(filename=file_path, batch_size=[n_samples_*n_corr*n_rw], mode = 'w')

            self._dss[cdsk]['data'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_rw,)+data_shape), dtype=torch.float32)
            self._dss[cdsk]['output'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_rw,)+os), dtype=torch.float32)
            self._dss[cdsk]['residual'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_rw,)+os), dtype=torch.float32)
            self._dss[cdsk]['latent_space'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_rw,)+ls), dtype=torch.float32)
            self._dss[cdsk]['corruption'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_rw,)), dtype=torch.int)
            self._dss[cdsk]['detection'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_rw,)), dtype=torch.int)                    
            self._dss[cdsk]['RW'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_rw,)), dtype=torch.int)
            self._dss[cdsk]['RW0'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_rw,)), dtype=torch.int)
            self._dss[cdsk]['RW1'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_rw,)), dtype=torch.int)
            self._dss[cdsk]['RW2'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_rw,)), dtype=torch.int)
            self._dss[cdsk]['RW3'] = MMT.empty(shape=torch.Size((n_samples_*n_corr*n_rw,)), dtype=torch.int)
            
            # Close PTD create with mode 'w' and re-open it with mode 'r+'
            # This is done so we can use multiple workers with the dataloaders 
            self._dss[cdsk].close()
            self._dss[cdsk] = PersistentTensorDict.from_h5(file_path, mode='r+')

            sentinel_dl = DataLoader(
                    dataset = self._dss[cdsk],
                    batch_size = n_samples_,
                    collate_fn = lambda x:x,
                    shuffle = False,
                    num_workers = n_threads
                    )
            
            for sentinel, config in zip(sentinel_dl, configs):
                group = config['group_id']
                chs = config['channels']
                corr = config['corruption']
                corr_id = config['corr_id']

                t = self._dss[loader]['data'][idx].clone()

                for c in chs:
                    sig = self._dss[loader]['data'][idx, 0, c, :].detach().cpu().numpy()
                    corr.fit(sig)
                    corr_sig = corr.distort(sig) 
                    t[:, 0, c, :] = torch.from_numpy(corr_sig).to(self._dss[loader].device)

                    sentinel['data'] = t

                sentinel['corruption'] = torch.ones(len(t))*corr_id
                sentinel['RW'] = torch.ones(len(t))*group
                sentinel[f'RW{group}'] = torch.ones(len(t))

            sentinel_dl = DataLoader(
                    dataset = self._dss[cdsk],
                    batch_size = bs,
                    collate_fn = lambda x:x,
                    shuffle = False,
                    num_workers = n_threads
                    )
        
            for sentinel in tqdm(sentinel_dl):
                with torch.no_grad():
                    x = sentinel['data'].float().to(model.device)
                    with torch.no_grad():
                        y, ls = model(x)

                    sentinel['output'] = y
                    sentinel['residual'] = y - x
                    sentinel['latent_space'] = ls

                    mse = mse_loss(y, x, reduction='none')  
                    mse_per_sample = mse.view(mse.size(0), -1).mean(dim=1)
                    sentinel['detection'] = (mse_per_sample > thr).long()
            
        return   
    
    def load_only(self, **kwargs):
        '''
        Load already computed dataset.

        Args:
        - loadrs (list[str]): load the specified loaders
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
        
            # data file path is the one which was converted to dss above
            _dfp = self.path/('dss.'+ds_key)

            if verbose: print(f'Loading files {_dfp} from disk. ')
            self._dss[ds_key] = PersistentTensorDict.from_h5(_dfp, mode=mode)
            #print(f'Loaded from h5:{self._dss[ds_key]['data'].dtype}') #i get NONE ?!!!!
            _n_samples = len(self._dss[ds_key])
            if verbose: print('loaded n_samples: ', _n_samples)
            
        return
