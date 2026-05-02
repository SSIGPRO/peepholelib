import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.parsedDataset import _ShardedPTD
from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import tempfile

if __name__ == '__main__':
    torch.manual_seed(0)
    n = 3  # samples per shard

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        shards = []
        for si in range(2):
            path = tmp / f'shard{si}'
            ptd = PTD(filename=path, batch_size=[n], mode='w')
            ptd['v'] = MMT.empty(shape=(n,), dtype=torch.float64)
            ptd.close()
            ptd = PTD.from_h5(path, mode='r+')
            ptd.batch_size = torch.Size((n,))

            dl = DataLoader(ptd, batch_size=n, collate_fn=lambda x: x)
            for d in dl:
                d['v'] = torch.randn(n, dtype=torch.float64)

            shards.append(ptd)

        print(shards)
        print(shards[0]['v'])
        print(shards[1]['v'])

        test = torch.cat([s['v'] for s in shards], dim=0)
        print(test)
        
        # shard 0: (global indices 0-2)
        # shard 1: (global indices 3-5)
        ds = _ShardedPTD(shards)

        # indices cross shards in mixed order → non-trivial permutation
        indices = [
            4, # shard 1, local index 1
            1, # shard 0, local index 1
            5, # shard 1, local index 2
            2, # shard 0, local index 2
            3  # shard 1, local index 0
            ]
        ## expected shard_groups = {1: ([0, 2, 4], [1, 2, 0]), 0: ([1, 3], [1, 2])}

        ## expected flat_positons = [0, 2, 4, 1, 3] 
        result = ds.__getitems__(indices)

        assert torch.all(result['v'] == test[indices]), f'FAIL: got {result["v"].tolist()}, expected {test[indices]}'
        print(f'PASS: __getitems__({indices}) -> {result["v"].tolist()}')

        for s in shards:
            s.close()
