# our stuff
from .classifier_base import DrillBase

# torch stuff
# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

# python stuff
from pathlib import Path 
import clip
from tqdm import tqdm
   
def get_clip_embeddings(self, **kwargs):
   '''
   Compute CLIP embeddings for all datasets in self._dss.

   Args:
   - batch_size (int): batch size for processing data. Defaults to 64.
   - n_threads (int): 'num_workers' passed to 'torch.utils.data.DataLoader'. Defaults to 1.
   - verbose (bool): print progress messages. Defaults to False.
   '''

   self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

   self._model, _ = clip.load(self.name, device=self.device)
   self._embed_shape = self._model.visual.proj.shape

   bs = kwargs.get('batch_size', 64) 
   n_threads = kwargs.get('n_threads', 1)
   verbose = kwargs.get('verbose', False)

   self._corevds = {}
   for ds_key in self._dss:
      #------------------------------------------------
      # pre-allocate clip embeddings
      #------------------------------------------------
      if verbose: print(f'\n ---- Getting CLIP embeddings for {ds_key}\n')
      file_path = self.path/(self.name.replace('/', '_')+'.'+ds_key)

      if file_path.exists():
         if verbose: print(f'File {file_path} exists. Loading from disk.')
         self._corevds = PersistentTensorDict.from_h5(file_path, mode='r+')
         n_samples = len(self._corevds)
         if verbose: print('loaded n_samples: ', n_samples)
         self._corevds.batch_size = torch.Size((n_samples,)) 
      else:
         n_samples = len(self._dss[ds_key])
         self._corevds = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
         if verbose: print('created clip embeddings with n_samples: ', n_samples)
      
      if verbose: print(f'\n ---- Getting CLIP embeddings for {ds_key}\n')

      self._corevds[ds_key] = MMT.empty(shape=((n_samples,)+self._embed_shape))
      self._corevds[ds_key].close()
      self._corevds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
      
      embeds_dl = DataLoader(self._corevds[ds_key], batch_size=bs, collate_fn=lambda x: x, shuffle=False, num_workers = n_threads)

      ds_dl = DataLoader(self._dss[ds_key], batch_size=bs, collate_fn=lambda x: x, shuffle=False, num_workers = n_threads)

      for embeds_data, ds_data in tqdm(zip(embeds_dl, ds_dl), disable=not verbose, total=len(embeds_dl)):

            with torch.no_grad():       
                    embeds_data = self._model.encode_image(ds_data['image'].to(self.device))