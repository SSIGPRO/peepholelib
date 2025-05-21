# General python stuff
from tqdm import tqdm
from functools import partial
from math import ceil

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

# Our stuff
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.datasets.dataset_base import DatasetBase 
from peepholelib.coreVectors.parsers import from_dataset
from peepholelib.coreVectors.prediction_fns import multilabel_classification
    
def parse_ds(self, **kwargs):
    self.check_uncontexted()
    
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    ds = kwargs['datasets']
    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 64
    n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 1 

    key_list = kwargs['key_list'] if 'key_list' in kwargs else ['image', 'label']
    ds_parser = kwargs['ds_parser'] if 'ds_parser' in kwargs else from_dataset 
    pred_fn = kwargs['pred_fn'] if 'pred_fn' in kwargs else multilabel_classification

    model = self._model
    num_classes = self._model.num_classes
    device = self._model.device 
    hooks = model.get_hooks()
    
    assert(isinstance(ds, DatasetBase))

    for ds_key in ds._dss:
        if verbose: print(f'\n ---- Getting data from {ds_key}\n')
        file_path = self.path/(self.name+'.dss.'+ds_key)

        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            self._n_samples[ds_key] = len(self._dss[ds_key])
            
            n_samples = self._n_samples[ds_key]
            if verbose: print('loaded n_samples: ', n_samples)
        else:
            self._n_samples[ds_key] = len(ds._dss[ds_key])
            n_samples = self._n_samples[ds_key] 
            if verbose: print('created datasets dict with n_samples: ', n_samples)
            self._dss[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
            
            #------------------------
            # Pre-allocation 
            #------------------------
            # dry parser run to get shapes
            if verbose: print(f'Allocating {key_list}')
            _data = [ds._dss[ds_key][0]]
            data = ds_parser(_data)

            for key in key_list:
                _d = data[key][0]
                # pre-allocation activations
                if _d.shape == torch.Size([]):
                    self._dss[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,))) 
                else:
                    self._dss[ds_key][key] = MMT.empty(shape=torch.Size((n_samples,)+_d.shape))
             
            if verbose: print(f'Allocating output, pred, result')
            # allocate memory for pred and result
            self._dss[ds_key]['output'] = MMT.empty(shape=torch.Size((n_samples,num_classes)))
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
            dl_ori = DataLoader(dataset=ds._dss[ds_key], batch_size=bs, collate_fn=partial(ds_parser), shuffle=False) 
            dl_dst = DataLoader(self._dss[ds_key], batch_size=bs, collate_fn=lambda x:x, shuffle=False, num_workers=n_threads)

            if verbose: print('Parsing dataset')
            for data_in, data_t in tqdm(zip(dl_ori, dl_dst), disable=not verbose, total=ceil(n_samples/bs)): 
                for key in key_list:
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

