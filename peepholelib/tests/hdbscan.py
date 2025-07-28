import cuml
cuml.accel.install()

from tensordict import PersistentTensorDict as PTD
from tensordict import MemoryMappedTensor as MMT 
import torch
from cuml.cluster.hdbscan import HDBSCAN, membership_vector, approximate_predict
from pathlib import Path
from torch.utils.data import DataLoader as DL
from matplotlib import pyplot as plt

if __name__ == '__main__':
    ds = 2
    ns = 200
    nc = 3
    eps = 0.005
    alpha = 2
    f = Path.cwd()/'banana'
    print('file: ', f)
    
    if f.exists(): 
        print('loading data')
        d = PTD.from_h5(filename=f, mode='r+')
    else:
        print('creating data')
        d = PTD(filename=f.as_posix(), batch_size=[nc*ns], mode='w')
        d['x'] = MMT.empty(shape=(nc*ns, ds))
        d['y'] = MMT.empty(shape=(nc*ns,))
    
        i = 1
        dl = DL(d, batch_size=ns, collate_fn=lambda x:x)
        for _d in dl:
            theta = torch.rand(ns)*torch.pi*2
            c = torch.cos(theta)
            s = torch.sin(theta)
            _d['x'] = alpha*i*torch.vstack((c, s)).T + eps*torch.rand(ns, ds)
            _d['y'] = i*torch.ones(ns)
            i += 1

    print('cuml: ', cuml.accel.enabled())
    
    clusterer = HDBSCAN(
            alpha = 1.0,
            min_cluster_size = 2,
            max_cluster_size = ns, # we know that
            cluster_selection_method = 'eom', #'leaf'
            #allow_single_cluster = True,
            #p = 2,
            prediction_data = True 
            )

    with cuml.accel.profile():
        clusterer.fit(X=d['x'].detach().numpy(), y=d['y'].detach().int().numpy(), convert_dtype=False)
        clusterer.generate_prediction_data()
        ncl = len(torch.unique(torch.tensor(clusterer.labels_)))

    tx = torch.zeros(nc*ns, ds)
    ty = torch.zeros(nc*ns)
    for i in range(nc):
        theta = torch.rand(ns)*torch.pi*2
        c = torch.cos(theta)
        s = torch.sin(theta)
        tx[i*ns:(i+1)*ns] = alpha*(i+1)*torch.vstack((c, s)).T + eps*torch.rand(ns, ds)
        ty[i*ns:(i+1)*ns] = i*torch.ones(ns)
 
    print('predicting')
    with cuml.accel.profile():
        pprob = membership_vector(
                clusterer = clusterer,
                points_to_predict = tx.detach().numpy(),
                batch_size = ns,
                convert_dtype = False,
                )
        py, _ = approximate_predict(
                clusterer = clusterer,
                points_to_predict = tx.detach().numpy(),
                convert_dtype = False,
                )
    
    # for the ones which fail, assing a cluster anyways
    # Comment the next line if you want to only take the succefully clustered ones
    py[py == -1] = pprob[py == -1].argmax(axis=1)
    #print(len(py), (py == -1).sum())
    # those probs are not normalized
    #print(pprob[py!=-1].sum(axis=1))

    plt.figure()
    plt.scatter(
            x = tx[py != -1, 0],
            y = tx[py != -1, 1],
            c = py[py != -1],
            #edgecolors = ty
            )
    plt.savefig('banana.png')
    plt.show()
    plt.close()
