from pathlib import Path
import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

def get_pcas(self, **kwargs):
    path = Path(kwargs.get('path'))
    name = kwargs.get('name')
    target_modules = kwargs.get('target_modules')
    components = kwargs.get('components', 100)    #k is the number of PCA components
    sample_in = kwargs.get('sample_in')
    verbose = kwargs.get('verbose')

    # create folder
    path.mkdir(parents=True, exist_ok=True)

    file_path = path/name
    # check if file exists
    if file_path.exists():
        if verbose: print(f'File{file_path} exits. Loading from disk.')
        _pcas = TensorDict.load_memmap(file_path)
    else:
        _pcas = TensorDict()

    #_modules_to_compute = []
    #for mk in target_modules:
    #    if mk in _pcas.keys():
    #        continue
    #    _modules_to_compute.append(mk)
    #if verbose: print(f'modules to compute PCA: {_modules_to_compute}')

    #turn on the act saving
    self.set_activations(save_input = True, save_output = False)

    latent_act = sample_in.to('cpu')
    #for mk in _modules_to_compute:
    if verbose: print(f'\nGetting PCA \n')

        #act = self._acts['in_activations'][mk]
    latent_mean = latent_act.mean(dim=0, keepdim=True)

    latent_centered = latent_act - latent_mean

    # Compute SVD for PCA
    U, S, Vt = torch.linalg.svd(latent_centered, full_matrices=False)
    #U, S, Vt = U.cpu().clone(), S.cpu().clone(), Vt.cpu().clone()

    # pre-allocate
    #Vt_mmt = MMT.empty((components, latent_act.shape[1]), dtype=torch.float32)
    #S_mmt = MMT.empty((components,), dtype=torch.float32)

    #Vt_mmt[:] = Vt[:components]
    #S_mmt[:] = S[:components]

    # keep only k-components
    #_pcas = TensorDict(
    #    {
    #        'Vt' : Vt_mmt,
    #        'S' : S_mmt
    #    }
    #)

      # Turn off activation saving
    #self.set_activations(save_input=False, save_output=False)

    if verbose: print(f'saving {file_path}')
    #if len(_modules_to_compute) != 0:
    #    _pcas.memmap(file_path)
    
    #self._pcas = _pcas
    # Keep top-k components
    Vt_top = Vt[:components].clone()
    S_top = S[:components].clone()

    # store results
    self._pcas = {
        'latent_space' : {
            'Vt': Vt_top, 
            'S': S_top
            }
        }

    if verbose:
        print(f'PCA Vt shape: {Vt_top.shape}, S shape: {S_top.shape}')

    return self._pcas