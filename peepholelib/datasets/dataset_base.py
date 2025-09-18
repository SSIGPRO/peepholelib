# General python stuff
from pathlib import Path as Path
import abc  
from tqdm import tqdm

# torch stuff
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

class DatasetBase(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        self.data_path = Path(kwargs.get('data_path'))

        # computed in __load_data__()
        self.__dataset__ = None # this one saves the dataset as given

        # computed in load_data()
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
    def load_data(self, **kwargs):
        # TODO: rework
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

        #self.check_uncontexted()
        
        save_path = Path(kwargs.get('save_path'))
        dss = kwargs.get('datasets')
        ds_parsers = kwargs.get('ds_parsers') ## it will be a dictionary
        ds_kwargs = kwargs.get('ds_kwargs', None) ## it will be a dictionary
        pred_fn = kwargs.get('pred_fn', multilabel_classification)

        bs = kwargs.get('batch_size', 64) 
        n_threads = kwargs.get('n_threads', 1) 

        verbose = kwargs.get('verbose', False) 
        
        # some defs for simplicity
        model = self._model
        device = self._model.device 

        if ds_kwargs == None:
            for k in dss:
                ds_kwargs[k] = {}

        ret = cls(data_path = save_path)
        ret._dss = {}

        for ds_name in dss:
            ds_parser = ds_parsers[ds_name]
            ds_kwargs = ds_kwargs[ds_name]
            dss[ds_name].__load_data__(**ds_kwargs)

            for ds_key in dss[ds_name].__datase__:
                if verbose: print(f'\n ---- Getting data from {ds_key}\n')
                file_path = ret.data_path/('dss.'+ds_key)

                if file_path.exists():
                    if verbose: print(f'File {file_path} exists. Loading from disk.')
                    ret._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

                    n_samples = len(ret._dss[ds_key])
                    
                    if verbose: print('loaded n_samples: ', n_samples)
                else:
                    n_samples = len(dss[ds_name].__dataset__[ds_key])
                    if verbose: print('creating dataset with n_samples: ', n_samples)
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

                    # TODO: not all datasets have labels???
                    if 'label' in data.keys():
                        ret._dss[ds_key]['result'] = MMT.empty(shape=torch.Size((n_samples,)))

                    #------------------------
                    # copy images and labels
                    #------------------------
                    # create dataloader of input dataset
                    dl_ori = DataLoader(
                            dataset = dss[ds_name].__dataset__[ds_key],
                            batch_size = bs,
                            collate_fn = parser,
                            shuffle = False
                            ) 

                    dl_dst = DataLoader(
                            ret._dss[ds_key],
                            batch_size=bs,
                            collate_fn=lambda x:x,
                            shuffle=False,
                            num_workers=n_threads
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

                            # TODO: not all datasets have labels???
                            if 'label' in data.keys():
                                data_t['result'] = predicted_labels == data_t['label']

        return ret

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


