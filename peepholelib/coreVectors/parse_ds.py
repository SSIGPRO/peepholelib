# General python stuff
from tqdm import tqdm
from math import ceil
from functools import partial

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
    '''
    Parse dataset, saving images, labels, model output, 'result' (1 if samples are correctly classified, 0 otherwise). I know, copying images and labels is redundant, but it is convenient to have them all in a common structure for the downstream computations.
    Data is saved into a 'tensordict.PersistentTensorDict' at 'self.path/dss.<loader>' (see 'coreVectors.__init__()'), with 'loader' being the loaders keys (see peepholelib.datasets). Alreday existing files are skipped.
    Args:
    - datasets (peepholelib.dataset_base.DatasetBase): Dataset wrapped in DatasetBase
    - batch_size (int): Creates dataloader to do computation in batch size. Defaults to 64.
    - n_threads (int): 'num_workers' passed to 'torch.utils.data.DataLoader'. Defaults to 1.
    - ds_parser (callable): Function taking batched dataset samples and parsing into a dictionary with keys = ['images', 'labels'].
    - pred_fn (callable): Function taking batched model's outputs and selecting a class.
    - verbose (bool): print progress messages.
    '''
    self.check_uncontexted()
    
    ds = kwargs['datasets']
    bs = kwargs.get('batch_size', 64) 
    n_threads = kwargs.get('n_threads', 1) 

    ds_parser = kwargs.get('ds_parser', from_dataset) 
    pred_fn = kwargs.get('pred_fn', multilabel_classification)

    verbose = kwargs.get('verbose', False) 

    model = self._model
    device = self._model.device 
    
    assert(isinstance(ds, DatasetBase))

    self._dss = {}
    for ds_key in ds._dss:
        if verbose: print(f'\n ---- Getting data from {ds_key}\n')
        file_path = self.path/('dss.'+ds_key)

        if file_path.exists():
            if verbose: print(f'File {file_path} exists. Loading from disk.')
            self._dss[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')

            n_samples = len(self._dss[ds_key])
            
            if verbose: print('loaded n_samples: ', n_samples)
        else:
            n_samples = len(ds._dss[ds_key])
            if verbose: print('created datasets dict with n_samples: ', n_samples)
            self._dss[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
            
            #------------------------
            # Pre-allocation 
            #------------------------
            # if verbose: print(f'Allocating {key_list}')

            # dry run to get shapes
            data = ds_parser(ds.get(ds_key,0))

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
            dl_ori = DataLoader(dataset=ds._dss[ds_key], batch_size=bs, collate_fn=ds_parser, shuffle=False) 
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

