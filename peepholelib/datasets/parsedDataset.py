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
    def parse_ds(cls, **kwargs):
        '''
        Parse datasets, saving images, labels, model output, 'result' (1 if samples are correctly classified, 0 otherwise). I know, copying images and labels is redundant, but it is convenient to have them all in a common structure for the downstream computations.
        Data is saved into a 'tensordict.PersistentTensorDict' at 'save_path/dss.<loader>', with 'loader' being the loaders keys (see peepholelib.datasets). Alreday existing files are skipped.
        Args:
        - datasets (peepholelib.dataset_base.DatasetBase): Dictionary with key being the name, and value an instance of specific dataset inheriting `datasets.DatasetBase`.
        - ds_parsers (dict(str: callable)): Dictionary with same keys as `datasets`, and values being functions taking batched dataset samples and parsing into a dictionary with keys = ['images', 'labels']. 
        - ds_samplers (dict(str: dict())): Dictionary with same keys as `datasets`, and values being a sampler (see `datasets.functional.samplers`). Facultative.
        - batch_size (int): Creates dataloader to do computation in batch size. Defaults to 64.
        - pred_fn (callable): Function taking batched model's outputs and selecting a class.
        - batch_size (int): Creates dataloader to do computation in batch size. Defaults to 64.
        - n_threads (int): 'num_workers' passed to 'torch.utils.data.DataLoader'. Defaults to 1.
        - verbose (bool): print progress messages.
        '''
        
        save_path = Path(kwargs.get('save_path'))
        model = kwargs.get('model')
        dss = kwargs.get('datasets')
        ds_parsers = kwargs.get('ds_parsers') ## it will be a dictionary
        ds_kwargs = kwargs.get('ds_kwargs', None) ## it will be a dictionary
        ds_samplers = kwargs.get('ds_samplers', None) ## it will be a dictionary
        pred_fn = kwargs.get('pred_fn', multilabel_classification)

        bs = kwargs.get('batch_size', 64) 
        n_threads = kwargs.get('n_threads', 1) 

        verbose = kwargs.get('verbose', False) 

        # some defs for simplicity
        device = model.device 

        save_path.mkdir(parents=True, exist_ok=True)
        ret = cls(path = save_path)
        ret._dss = {}

        # enter the context manager
        with ret:
            for ds_name in dss:
                ds_parser = ds_parsers[ds_name]
                
                dss[ds_name].__load_data__()

                if ds_samplers != None and ds_name in ds_samplers:
                    if verbose: print(f'Applying {ds_samplers[ds_name]} to {ds_name}')
                    ds_samplers[ds_name](ds = dss[ds_name])

                for ds_key in dss[ds_name].__dataset__:
                    if verbose: print(f'\n ---- Getting data from {ds_key}\n')
                    file_path = ret.path/('dss.'+ds_key)
                    
                    if file_path.exists():
                        if verbose: print(f'File {file_path} exists. Loading from disk.')
                        ret._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

                        n_samples = len(ret._dss[ds_key])
                        
                        if verbose: print('loaded n_samples: ', n_samples)
                    else:
                        n_samples = len(dss[ds_name].__dataset__[ds_key])
                        if verbose: print('Creating dataset with n_samples: ', n_samples)
                        ret._dss[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
                        
                        #------------------------
                        # Pre-allocation 
                        #------------------------
                        # if verbose: print(f'Allocating {key_list}')

                        # dry run to get shapes
                        data = ds_parser(dss[ds_name].get(ds_key,0))

                        with torch.no_grad():
                            _res = model(data['image'].to(device))
                            num_classes = _res.shape[1]

                        for key in data.keys():
                            _d = data[key]
                            # pre-allocation activations
                            ret._dss[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,)+_d.shape[1:]))
                         
                        if verbose: print(f'Allocating output, pred, result')
                        # allocate memory for pred and result
                        ret._dss[ds_key]['output'] = MMT.empty(shape=torch.Size((n_samples, num_classes)))
                        ret._dss[ds_key]['pred'] = MMT.empty(shape=torch.Size((n_samples,)))
                        ret._dss[ds_key]['result'] = MMT.empty(shape=torch.Size((n_samples,)))

                        # Close PTD create with mode 'w' and re-open it with mode 'r+'
                        # This is done so we can use multiple workers with the dataloaders 
                        ret._dss[ds_key].close()
                        ret._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

                        #------------------------
                        # copy images and labels
                        #------------------------
                        # create dataloader of input dataset
                        dl_ori = DataLoader(
                                dataset = dss[ds_name].__dataset__[ds_key],
                                batch_size = bs,
                                collate_fn = ds_parser,
                                shuffle = False
                                ) 

                        dl_dst = DataLoader(
                                ret._dss[ds_key],
                                batch_size = bs,
                                collate_fn = lambda x:x,
                                shuffle = False,
                                num_workers = n_threads
                                )

                        if verbose: print(f'Parsing {ds_key}')
                        for data_in, data_t in tqdm(zip(dl_ori, dl_dst), disable=not verbose, total=ceil(n_samples/bs)): 
                            for key in data_in.keys():
                                
                                data_t[key] = data_in[key]
                        
                            # ---------------------------------------
                            # compute predictions and get activations
                            # ---------------------------------------
                            with torch.no_grad():
                                y_predicted = model(data_t['image'].to(device))
                        
                                predicted_labels = pred_fn(y_predicted).detach().cpu()
                                data_t['output'] = y_predicted
                                data_t['pred'] = predicted_labels
                                data_t['result'] = predicted_labels == data_t['label']
        
        return ret

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


