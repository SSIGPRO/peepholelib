# General python stuff
from pathlib import Path as Path
import abc  
from tqdm import tqdm

# torch stuff
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

class DatasetBase(metaclass=abc.ABCMeta):

    from .parse_ds import parse_ds

    def __init__(self, **kwargs):
        name = kwargs.get('name')
        self.data_path = Path(kwargs.get('data_path'), None)

        # computed in load_data()
        self.__dataset__ = None # this one saves the dataset as given
        self._dss = None # this is the parsed datasets as PTD
        self._classes = None
    
    @abc.abstractmethod
    def __load_data__(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get(self):
        raise NotImplementedError()
    
    def get_classes(self):
        if not self._classes:
            raise RuntimeError('Data not loaded. Please run model.load_data() first.')

        return self._classes
   
    @classmethod
    def soft_stack(cls, insts):
        ret = cls()
        for inst int insts:
            for ds_key in inst._dss:
                ret._dss[ds_key] = inst._dss[ds_key]
                if inst._classes != None:
                    ret._classes[ds_key] = inst._classes[ds_key]
        return ret
    
    def load_data(self):
        '''
        Parse dataset, saving images, labels, model output, 'result' (1 if samples are correctly classified, 0 otherwise). I know, copying images and labels is redundant, but it is convenient to have them all in a common structure for the downstream computations.
        Data is saved into a 'tensordict.PersistentTensorDict' at 'self.path/dss.<loader>' (see 'DatasetBase.__init__()'), with 'loader' being the loaders keys (see peepholelib.datasets). Alreday existing files are skipped.
        Args:
        - datasets (peepholelib.dataset_base.DatasetBase): Dataset wrapped in DatasetBase
        - batch_size (int): Creates dataloader to do computation in batch size. Defaults to 64.
        - n_threads (int): 'num_workers' passed to 'torch.utils.data.DataLoader'. Defaults to 1.
        - ds_parser (callable): Function taking batched dataset samples and parsing into a dictionary with keys = ['images', 'labels'].
        - pred_fn (callable): Function taking batched model's outputs and selecting a class.
        - verbose (bool): print progress messages.
        '''
        self.check_uncontexted()
        
        bs = kwargs.get('batch_size', 64) 
        n_threads = kwargs.get('n_threads', 1) 

        ds_parser = kwargs.get('ds_parser') ## it will be a dictionary
        pred_fn = kwargs.get('pred_fn', multilabel_classification)

        verbose = kwargs.get('verbose', False) 

        model = self._model
        device = self._model.device 

        self._dss = {}
        for ds_key, parser in ds_parser.items():
            if verbose: print(f'\n ---- Getting data from {ds_key}\n')
            file_path = self.path/('dss.'+ds_key)

            if file_path.exists():
                if verbose: print(f'File {file_path} exists. Loading from disk.')
                self._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

                n_samples = len(self._dss[ds_key])
                
                if verbose: print('loaded n_samples: ', n_samples)
            else:
                n_samples = len(self.dss_[ds_key])
                if verbose: print('created datasets dict with n_samples: ', n_samples)
                self._dss[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
                
                #------------------------
                # Pre-allocation 
                #------------------------
                # if verbose: print(f'Allocating {key_list}')

                # dry run to get shapes
                data = parser(ds.get(ds_key,0))

                with torch.no_grad():
                    _res = model(data['image'].to(device))
                    num_classes = _res.shape[1]

                for key in data.keys():
                    _d = data[key]
                    # pre-allocation activations
                    self._dss[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,)+_d.shape[1:]))
                 
                if verbose: print(f'Allocating output, pred, result')
                # allocate memory for pred and result
                self._dss[ds_key]['output'] = MMT.empty(shape=torch.Size((n_samples, num_classes)))
                self._dss[ds_key]['pred'] = MMT.empty(shape=torch.Size((n_samples,)))
                self._dss[ds_key]['result'] = MMT.empty(shape=torch.Size((n_samples,)))

                # Close PTD create with mode 'w' and re-open it with mode 'r+'
                # This is done so we can use multiple workers for reading and writting
                self._dss[ds_key].close()
                self._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

                #------------------------
                # copy images and labels
                #------------------------
                # create dataloader of input dataset and activations
                dl_ori = DataLoader(dataset=self.dss_[ds_key], batch_size=bs, collate_fn=parser, shuffle=False) 
                dl_dst = DataLoader(self._dss[ds_key], batch_size=bs, collate_fn=lambda x:x, shuffle=False, num_workers=n_threads)

                if verbose: print('Parsing dataset')
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

        return

    def load_only(self, **kwargs):
        # TODO: think about it
        '''
        Load already computed dataset.

        Args:
        - loadrs (list[str]): load the specified loaders
        - mode (str): Opens the file with the specified mode. See 'tensordict.PersistentTensorDict.from_h5()' for details. Defaults to 'r'.
        - norm_file (str): load the normalization information. Defaults to None. 
        - verbose (bool): print progress messages.
        '''
        self.check_uncontexted()

        loaders = kwargs.get('loaders')
        mode = kwargs.get('mode', 'r')
        norm_file = kwargs.get('norm_file', None)
        if norm_file != None: norm_file = Path(norm_file)
        verbose = kwargs.get('verbose', False)

        self._dss = {}
        for ds_key in loaders:
            if verbose: print(f'\n ---- Getting data from {ds_key}\n')
            
            _dss_file_paths = self.data_path/('dss.'+ds_key)

            if verbose: print(f'Loading files {_dss_file_paths} from disk. ')
            self._dss[ds_key] = PersistentTensorDict.from_h5(_dss_file_paths, mode=mode)

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


