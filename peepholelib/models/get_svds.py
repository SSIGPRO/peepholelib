# General python stuff
from pathlib import Path

# torch stuff
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

def get_svds(self, **kwargs):
    path = Path(kwargs.get('path'))
    name = kwargs.get('name')
    target_modules = kwargs.get('target_modules')
    svd_fns = kwargs.get('svd_fns')
    sample_in = kwargs.get('sample_in')
    verbose = kwargs.get('verbose', False)

    # create folder
    path.mkdir(parents=True, exist_ok=True)
    
    file_path = path/name
    if file_path.exists():
        if verbose: print(f'File {file_path} exists. Loading from disk.')
        _svds = TensorDict.load_memmap(file_path)
    else: 
        _svds = TensorDict()

    _modules_to_compute = []
    for mk in target_modules:
        if mk in _svds.keys():
            continue
        _modules_to_compute.append(mk)
    if verbose: print('modules to compute SVDs: ', _modules_to_compute)
   
    # Turn on activation saving
    self.set_activations(save_input=True, save_output=False)
    
    # Dry run to get shapes
    with torch.no_grad():
        print('super dummy', sample_in.shape)
        _in = sample_in.reshape((1,)+sample_in.shape).to(self.device)
        print('extra dummy', _in.shape)
        self(_in)
        print('TI PREGOOO FUNzIONAAA')

    for mk in _modules_to_compute:
        if verbose: print(f'\n ---- Getting SVDs for {mk}\n')
        module = self._target_modules[mk]
        
        in_shape = self._acts['in_activations'][mk].shape[1:]
        U, s, Vh = svd_fns[mk](layer=module, in_shape=in_shape) 
        U, s, Vh = U.detach().cpu(), s.detach().cpu(), Vh.detach().cpu()
        _svds[mk] = TensorDict({
                'U': MMT(U),
                's': MMT(s),
                'Vh': MMT(Vh)
                })

    # Turn off activation saving
    self.set_activations(save_input=False, save_output=False)

    if verbose: print(f'saving {file_path}')
    if len(_modules_to_compute) != 0:
        _svds.memmap(file_path)
    
    self._svds = _svds
    return self._svds
