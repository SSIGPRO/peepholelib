# torch stuff
import torch

# tensordict
from tensordict import TensorDict

class _ShardedPTD:
    '''
    Read-only wrapper over multiple `PersistentTensorDict` shards that presents them as a single dataset. Shard files are named `dss.<ds_key>.chunk_<i>`.
    '''
    def __init__(self, shards):
        self.shards = shards
        self._keys = list(self.shards[0].keys())
        lengths = torch.tensor([0]+[len(s) for s in shards])
        self._cum = lengths.cumsum(dim=0)
        self._total = lengths.sum().item()
        return

    def _resolve(self, idx):
        if idx >= self._total or idx < -self._total: 
            raise IndexError(f'Index {idx} out of range for sharded dataset of size {self._total}')

        if idx < 0:
            idx = self._total + idx

        for i in range(len(self.shards)):
            if idx < self._cum[i + 1]:
                return i, (idx - self._cum[i]).item()
        
    def __len__(self):
        return self._total

    def keys(self):
        return self._keys

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx not in self._keys:
                raise KeyError(idx)
            return torch.cat([shard[idx] for shard in self.shards], dim=0)

        if isinstance(idx, int):
            si, li = self._resolve(idx)
            return self.shards[si][li]

        elif isinstance(idx, slice):
            idx = list(range(*idx.indices(self._total)))

        return self.__getitems__(idx)

    def __getitems__(self, indices):
        shard_groups = {}
        for pos, i in enumerate(indices):
            si, li = self._resolve(int(i))
            if si not in shard_groups:
                shard_groups[si] = ([], [])
            shard_groups[si][0].append(pos)
            shard_groups[si][1].append(li)

        parts = []
        flat_positions = []
        for si, (positions, local_idxs) in shard_groups.items():
            parts.append(self.shards[si][local_idxs])
            flat_positions.extend(positions)
        cat = torch.cat(parts, dim=0)

        return cat[torch.argsort(torch.tensor(flat_positions))]

    def close(self):
        for s in self.shards:
            s.close()
        return

class _StackedDS:
    '''
    This class is a wrap for multiple `PersistentTensorDicts` (PTDs) with DIFFERENT KEYS, oferring a common interface to emulate as if it was one PTD with all the keys. 
    '''
    def __init__(self, **kwargs):
        '''
        Args:
        - ori (tensordict.PersistentTensorDict | peepholelib.datasets.datasetWrap.DatasetWrap): Original dataset wrapped or a PTD. Both are supported since sampling from both return a dictionary.
        '''
        self.ori = kwargs['ori']
        self.transform = lambda x: x

        # set in stack_inference
        self.inf = None

        s_ori = self.ori[0]

        self.k_ori = list(s_ori.keys())
        self.k_inf = [] 
        self.len = len(self.ori)
        return
    
    def set_transform(self, transform=None):
        if transform != None:
            self.transform = transform
        return

    def stack_inference(self, inf):
        s_ori = self.ori[0]
        s_inf = inf[0]

        if len(self.ori) != len(inf): raise RuntimeError(f'Original dataset has {len(self.ori)} samples, parsed inference values have {len(inf)} samples. They should have the same number of samples.')

        # if keys appear in both, prefer inference values over original values
        self.k_ori = [k for k in s_ori.keys() if k not in s_inf.keys()]
        self.k_inf = list(s_inf.keys()) 

        # save the inf
        self.inf = inf

        return

    def __getitem__(self, idx):
        # this offers the ptd['key'] interface
        if type(idx) == str:
            if idx in self.k_ori:
                return self.ori[idx]
            elif idx in self.k_inf:
                return self.inf[idx]

        r = {}

        for k in self.k_ori:
            r[k] = self.ori[idx][k] 

        if self.inf != None:
            for k in self.k_inf:
                r[k] = self.inf[idx][k] 

        return self.transform(r) 
    
    def __getitems__(self, idx):

        r = TensorDict({}, batch_size=len(idx))

        ori_batch = self.ori.__getitems__(idx)
        for k in self.k_ori:
            r[k] = ori_batch[k]

        if self.inf != None:
            inf_batch = self.inf.__getitems__(idx)
            for k in self.k_inf:
                r[k] = inf_batch[k]

        return self.transform(r)

    def __len__(self):
        return self.len

    def keys(self):
        return self.k_ori + self.k_inf

    def close(self):
        self.ori.close()

        if self.inf != None:
            self.inf.close()
        return

class _ModuleWiseStack:
    def __init__(self, **kwargs):
        _tds = kwargs['tds']
        self.tds = {}

        # should have the same length
        lens = [len(td) for td in _tds.values()]
        if len(set(lens)) > 1:
            raise RuntimeError(f'PTDs should have the same lenght. Got: {lens}')

        self.tds = _tds

        self.len = lens[0]
        return

    def __getitem__(self, idx):
        if type(idx) == str:
            return self.tds[idx][idx]
        else:
            r = {}
            for k in self.tds.keys():
                r[k] = self.tds[k][k][idx] 
            return r 
    
    def __getitems__(self, idx):
        r = TensorDict({}, batch_size=len(idx))
        for k in self.tds.keys():
            r[k] = self.tds[k][k][idx] 

        return r 

    def __len__(self):
        return self.len

    def keys(self):
        return self.tds.keys()

    def mean(self, dim=0):
        r = TensorDict({})
        for k in self.tds.keys():
            r[k] = self.tds[k][k].mean(dim=dim) 
        return r
                                                     
    def std(self, dim=0):
        r = TensorDict({})
        for k in self.tds.keys():
            r[k] = self.tds[k][k].std(dim=dim) 
        return r                                    

    def close(self):
        for k in self.tds.keys():
            self.tds[k].close()
        return
