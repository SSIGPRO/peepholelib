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

   self.check_uncontexted()
    
   datasets = kwargs.get('datasets')
   
   bs = kwargs.get('batch_size', 64) 
   n_threads = kwargs.get('n_threads', 1) 

   verbose = kwargs.get('verbose', False) 

   device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
   clip_model = kwargs.get('clip_model', 'ViT-B/32')
   model, _ = clip.load(clip_model, device=device)
   embed_shape = model.visual.proj.shape

   self._corevds = {}
   for ds_key in datasets._dss:
      #------------------------------------------------
      # pre-allocate clip embeddings
      #------------------------------------------------
      if verbose: print(f'\n ---- Getting CLIP embeddings for {ds_key}\n')
      file_path = self.path/(self.name+'.'+ds_key)

      if file_path.exists():
         if verbose: print(f'File {file_path} exists. Loading from disk.')
         self._corevds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
         n_samples = len(self._corevds)
         if verbose: print('loaded n_samples: ', n_samples)
         self._corevds[ds_key].batch_size = torch.Size((n_samples,)) 
      else:
         n_samples = len(datasets._dss[ds_key])
         self._corevds[ds_key] = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')
         if verbose: print('created clip embeddings with n_samples: ', n_samples)
      
      if verbose: print(f'\n ---- Getting CLIP embeddings for {ds_key}\n')
      print(embed_shape)

      self._corevds[ds_key]['embedding'] = MMT.empty(shape=((n_samples,)+(embed_shape[1],)))

      self._corevds[ds_key].close()
      self._corevds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
      
      embeds_dl = DataLoader(self._corevds[ds_key], batch_size=bs, collate_fn=lambda x: x, shuffle=False, num_workers = n_threads)

      ds_dl = DataLoader(datasets._dss[ds_key], batch_size=bs, collate_fn=lambda x: x, shuffle=False, num_workers = n_threads)

      for embeds_data, ds_data in tqdm(zip(embeds_dl, ds_dl), disable=not verbose, total=len(embeds_dl)):

         with torch.no_grad():       
            embeds_data['embedding'] = model.encode_image(ds_data['image'].to(device))