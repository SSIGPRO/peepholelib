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
        
        path.mkdir(parents=True, exist_ok=True)
        cls_inst = cls(path = path)
        cls_inst._dss = {}

        ds_wrap.__load_data__()

        for ds_key in ds_wrap.__dataset__:
            file_path = cls_inst.path/('dss.'+ds_key)
            n_samples = len(ds_wrap.__dataset__[ds_key])
            
            # check if PTD exists 
            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                cls_inst._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            else:
                if verbose: print(f'Creating {ds_key} dataset with n_samples: ', n_samples)
                cls_inst._dss[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
                
                # get sample to get shapes
                sample = ds_wrap.__dataset__[ds_key][0]
                for key in sample.keys():
                    if verbose: print(f'allocating {key} with shape {sample[key].shape}')
                    cls_inst._dss[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,)+sample[key].shape), dtype=sample[key].dtype)
        
                # create Dataloader of input dataset
                ds_in = DataLoader(
                    dataset = ds_wrap.__dataset__[ds_key],
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
        return cls_inst
    
    def parse_ds(self, **kwargs):
        #self.check_uncontexted()

        model = kwargs['model']
        loaders = kwargs.get('loaders', None)
        bs = kwargs.get('batch_size', 2**11)
        verbose = kwargs.get('verbose', False)
        ds_samplers = kwargs.get('ds_samplers', None)
        pred_fn = kwargs.get('pred_fn', multilabel_classification)
        result_fn = kwargs.get('result_fn', results_one_hot_encoding)
        
        if loaders == None:
            loaders = self._dss.keys()

        for ds_key in loaders:
            n_samples = len(self._dss[ds_key])

            if ds_samplers is not None and ds_key in ds_samplers:
                if verbose:
                    print(f'Applying {ds_samplers[ds_key]} to {ds_key}')
                ds_samplers[ds_key](ds=self._dss[ds_key])
            
            # dataset sample for dry run
            sample = self._dss[ds_key][0:1]['image'].to(model.device)
            with torch.no_grad():
                _out, _ls = model(sample)
                     
            os = _out.shape[1:]
            ls = _ls.shape[1:]

            # check and skip if the values are already there
            if ('output' in self._dss[ds_key]) and ('pred' in self._dss[ds_key]) and ('result' in self._dss[ds_key]):
                continue

            # need to fix the batch size - workaround  
            self._dss[ds_key].batch_size = torch.Size((n_samples,))
            # allocate disk space
            num_classes =_out.shape[1]
            print("number of classes: ", num_classes)
            quit()
            self._dss[ds_key]['output'] = MMT.empty(shape=torch.Size((n_samples, num_classes)))
            self._dss[ds_key]['pred'] = MMT.empty(shape=torch.Size((n_samples,)))
            self._dss[ds_key]['result'] = MMT.empty(shape=torch.Size((n_samples,))) 

            dl = DataLoader(
                self._dss[ds_key],
                collate_fn = lambda x:x, 
                batch_size = bs
            )

            for data in tqdm(dl, disable=not verbose, total=ceil(n_samples/bs)):
                #compute predictions which is the out of decoder
                with torch.no_grad():
                    y_predicted = model(data['image'].to(model.device))
            
                    predicted_labels = pred_fn(y_predicted).detach().cpu()
                    data['output'] = y_predicted
                    data['pred'] = predicted_labels
                    data['result'] = result_fn(predicted_labels, data['label'])         
        
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


