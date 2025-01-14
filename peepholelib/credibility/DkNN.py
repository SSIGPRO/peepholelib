import numpy as np
from tempfile import mkdtemp
import os.path as path
import enum
import copy
from bisect import bisect_left
import warnings
import platform
import faiss
import falconn

import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

from pathlib import Path as Path

class NearestNeighbor:

    class BACKEND(enum.Enum):
        FALCONN = 1
        FAISS = 2

    def __init__(self, backend, dimension, neighbors, number_bits, nb_tables=None):
        assert backend in NearestNeighbor.BACKEND

        self._NEIGHBORS = neighbors
        self._BACKEND = backend

        if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
            self._init_falconn(dimension, number_bits, nb_tables)
        elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
            self._init_faiss(dimension,number_bits)
        else:
            raise NotImplementedError

    def _init_falconn(self, dimension, number_bits, nb_tables):
        assert nb_tables >= self._NEIGHBORS

        # LSH parameters
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = dimension
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = nb_tables
        params_cp.num_rotations = 2  # for dense set it to 1; for sparse data set it to 2
        params_cp.seed = 5721840
        params_cp.num_setup_threads = 0  # we want to use all the available threads to set up
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable

        # we build number_bits-bit hashes so that each table has
        # 2^number_bits bins; a rule of thumb is to have the number
        # of bins be the same order of magnitude as the number of data points
        falconn.compute_number_of_hash_functions(number_bits, params_cp)
        self._falconn_table = falconn.LSHIndex(params_cp)
        self._falconn_query_object = None
        self._FALCONN_NB_TABLES = nb_tables

    def _init_faiss(self, dimension, number_bits):
        # res = faiss.StandardGpuResources()
        # self._faiss_index = faiss.GpuIndexFlatL2(res, dimension)
        self._faiss_index = faiss.IndexLSH(dimension, number_bits)


    def add(self, x):
        if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
            self._falconn_table.setup(x)
        elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
            self._faiss_index.add(x)
        else:
            raise NotImplementedError

    def find_knns(self, x, output):
        if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
            return self._find_knns_falconn(x, output)
        elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
            return self._find_knns_faiss(x, output)
        else:
            raise NotImplementedError

    def _find_knns_falconn(self, x, output):
        # Late falconn query_object construction
        # Since I suppose there might be an error
        # if table.setup() will be called after
        if self._falconn_query_object is None:
            self._falconn_query_object = self._falconn_table.construct_query_object()
            self._falconn_query_object.set_num_probes(self._FALCONN_NB_TABLES)

        missing_indices = np.zeros(output.shape, dtype=bool)

        for i in range(x.shape[0]):
            query_res = self._falconn_query_object.find_k_nearest_neighbors(x[i], self._NEIGHBORS)

            try:
                output[i, :] = query_res
            except:
                # mark missing indices
                missing_indices[i, len(query_res):] = True
                output[i, :len(query_res)] = query_res

        return missing_indices

    def _find_knns_faiss(self, x, output):
        neighbor_distance, neighbor_index = self._faiss_index.search(x, self._NEIGHBORS)

        missing_indices = neighbor_distance == -1
        d1 = neighbor_index.reshape(-1)

        output.reshape(-1)[np.logical_not(missing_indices.flatten())] = d1[np.logical_not(missing_indices.flatten())]

        return missing_indices    
        
class DkNN:
   
    def __init__(self, **kwargs):
        """
        Implementation of the DkNN algorithm, see https://arxiv.org/abs/1803.04765 for more details
        :param model: model to be used
        :param nb_classes: the number of classes in the task
        :param neighbors: number of neighbors to find per layer
        :param layers: a list of layer names to include in the DkNN
        :param trainloader: data loader for the training data
        :param nearest_neighbor_backend: falconn or faiss to be used for LSH
        :param nb_tables: number of tables used by FALCONN to perform locality-sensitive hashing.
        :param number_bits: number of ha1.26.4          py311h24aa872_0  
numpy-base                1.26.4 sh bits used by LSH.
        """
        print('---------- DkNN init')
        print()

        self.model = kwargs['model']
        self.nb_classes = kwargs['nb_classes']
        self.neighbors = kwargs['neighbors']
        self.cv_dl = kwargs['cv_dl']
        self.percentage = kwargs['percentage']
        self.device = kwargs['device']
        self.seed = kwargs['seed']
        self.verbose = kwargs['verbose']
        self.path = Path(kwargs['path'])
        self.name = Path(kwargs['name'])
        self.backend = kwargs['nearest_neighbor_backend']
        self.nb_tables = kwargs['nb_tables']
        self.number_bits = kwargs['number_bits']

        self.nb_cali = -1
        self.calibrated = False
        self.res = TensorDict()

        layers = list(self.model._target_layers.keys()) 
        pt = self.percentage['train']
        pv = self.percentage['val']

        self.dknn_path = self.path/Path(f'{layers}/nb_tables_{self.nb_tables}/neighbor_{self.neighbors}/train_{pt}/val_{pv}')
        self.name = Path(self.name)

        if self.dknn_path.exists():
            if self.verbose: print(f'File {self.dknn_path} exists.')
            for ds_key in self.cv_dl:
                self.res[ds_key] = TensorDict.load_memmap(self.dknn_path/ds_key)
        else:
            if self.verbose: print(f'File {self.dknn_path} does not exists. Initializing class DkNN')
            self.dknn_path.mkdir(parents=True, exist_ok=True)
            portion = 'train'
            
            print('inizio labels')

            self.train_labels = self.cv_dl['train'].dataset['label']
            n_samples = len(self.train_labels)
            
            if self.percentage[portion] != 100:
        
                idx = np.arange(0, n_samples)
                rng = np.random.default_rng(self.seed)
                idx_rand = []
                
                for l in np.arange(0,self.nb_classes):
                    idx_l = np.argwhere(self.train_labels.numpy().astype(int)==l)
                    n_samples_l = len(idx_l)
                    num_elements_to_select = int(n_samples_l*(self.percentage[portion]/100))
                    rng.shuffle(idx_l)
                    idx_rand.append(idx_l[:num_elements_to_select])
                
                idx_rand = np.concatenate(idx_rand)
                
                self.train_labels = self.train_labels[idx_rand[:,0]]
                
            print("## Constructing the NearestNeighbor tables")
            self.query_objects = {} 
            self.centers = {}
            for layer in self.model._target_layers:
                filename = path.join(mkdtemp(), 'newfile.dat')
                filename = path.join(mkdtemp(), 'newfile.dat')
                act = np.memmap(filename, dtype='float32', mode='w+', shape=self.cv_dl['train'].dataset['in_activations'][layer].flatten(start_dim=1).shape)
                act[:] = self.cv_dl['train'].dataset['in_activations'][layer].flatten(start_dim=1)[:]
                
                # activations = self.cv_dl['train'].dataset['in_activations'][layer].flatten(start_dim=1).detach().cpu().numpy()
                if self.percentage[portion] != 100:
                    act = act[idx_rand[:,0]]
                    print(act.shape)
                
                # Build locality-sensitive hashing tables for training representations
                # train_activations_lsh = copy.copy(activations)
                 # mean of training data representation per layer (that needs to be substracted before NearestNeighbor)
                
                self.init_lsh(layer, act)
            

    def init_lsh(self, layer, train_activations_lsh):
        """
        Initializes locality-sensitive hashing with FALCONN to find nearest neighbors in training data
        """
        print("Constructing table for {}".format(layer))

        # Normalize all the lenghts, since we care about the cosine similarity
        train_activations_lsh /= np.linalg.norm(train_activations_lsh, axis=1).reshape(-1, 1)

        # Center the dataset and the queries: this improves the performance of LSH quite a bit
        center = np.mean(train_activations_lsh, axis=0)
        train_activations_lsh -= center
        self.centers[layer] = center

        # Constructing nearest neighbor table
        self.query_objects[layer] = NearestNeighbor(
            backend=self.backend,
            dimension=train_activations_lsh.shape[1],
            number_bits=self.number_bits,
            neighbors=self.neighbors,
            nb_tables=self.nb_tables,
        )

        self.query_objects[layer].add(train_activations_lsh)

        print("done!")
        print()   
    def calibrate(self, portion='val'):
        """
        Runs the DkNN on holdout data to calibrate the credibility metric
        :param calibloader: data loader for the calibration loader
        """
        print('---------- DkNN calibrate')
        print()
    
        # Compute calibration data activations
        self.cali_labels = self.cv_dl[portion].dataset['label']
        
        n_samples = len(self.cali_labels)
        
        if self.percentage[portion] != 100:
    
            idx = np.arange(0, n_samples)
            rng = np.random.default_rng(self.seed)
            idx_rand = []
            
            for l in np.arange(0,self.nb_classes):
            
                idx_l = np.argwhere(self.cali_labels.numpy().astype(int)==l)            
                n_samples_l = len(idx_l)
                num_elements_to_select = int(n_samples_l*(self.percentage[portion]/10))            
                rng.shuffle(idx_l)                
                idx_rand.append(idx_l[:num_elements_to_select])            
            
            idx_rand = np.concatenate(idx_rand,axis=0)
        
            self.cali_labels = self.cali_labels[idx_rand[:,0]]
        self.nb_cali = len(self.cali_labels)
        print("## Starting calibration of DkNN")
        knns_ind = {}
        knns_labels = {}
        for layer in self.model._target_layers:
            
            print(f"## calibration of {layer}")

            filename = path.join(mkdtemp(), 'newfile.dat')
            act = np.memmap(filename, dtype='float32', mode='w+', shape=self.cv_dl[portion].dataset['in_activations'][layer].flatten(start_dim=1).shape)
            act[:] = self.cv_dl[portion].dataset['in_activations'][layer].flatten(start_dim=1).detach().cpu().numpy()[:]
            
            if self.percentage[portion] != 100:
                act = act[idx_rand[:,0]]
                
            knns_ind[layer], knns_labels[layer] = self.find_train_knns(act, layer)

        
        assert all([v.shape == (self.nb_cali, self.neighbors) for v in knns_ind.values()])
        assert all([v.shape == (self.nb_cali, self.neighbors) for v in knns_labels.values()])
    
        cali_knns_not_in_class = self.nonconformity(knns_labels)
        cali_knns_not_in_l = np.zeros(self.nb_cali, dtype=np.int32)
        
        for i in range(self.nb_cali):
            cali_knns_not_in_l[i] = cali_knns_not_in_class[i, self.cali_labels[i].numpy().astype(int)]
    
        cali_knns_not_in_l_sorted = np.sort(cali_knns_not_in_l)
        self.cali_nonconformity = np.trim_zeros(cali_knns_not_in_l_sorted, trim='f')
        self.nb_cali = self.cali_nonconformity.shape[0]
        self.calibrated = True
    
    def find_train_knns(self, data_activations, layer):
        """
        Given a data_activation dictionary that contains a np array with activations for each layer,
        find the knns in the training data
        """
            
        # Pre-process representations of data to normalize and remove training data mean
       
        nb_data = data_activations.shape[0]
        data_activations /= np.linalg.norm(data_activations, axis=1).reshape(-1, 1)
        data_activations -= self.centers[layer]

        # Use FALCONN to find indices of nearest neighbors in training data
        knns_ind = np.zeros((data_activations.shape[0], self.neighbors), dtype=np.int32)
        knn_errors = 0
        print('starting finding knn')

        knn_missing_indices = self.query_objects[layer].find_knns(data_activations, knns_ind)
        knn_errors += knn_missing_indices.flatten().sum()

        # Find labels of neighbors found in the training data
        knns_labels = np.zeros((nb_data, self.neighbors), dtype=np.int32)

        knns_labels.reshape(-1)[
            np.logical_not(knn_missing_indices.flatten())
        ] = self.train_labels[
            knns_ind.reshape(-1)[np.logical_not(knn_missing_indices.flatten())]                    
        ]

        return knns_ind, knns_labels
    
    def nonconformity(self, knns_labels):
        """
        Given an dictionary of nb_data x nb_classes dimension, compute the nonconformity of
        each candidate label for each data point: i.e. the number of knns whose label is
        different from the candidate label
        """
        nb_data = knns_labels[list(self.model._target_layers)[0]].shape[0]
        knns_not_in_class = np.zeros((nb_data, self.nb_classes), dtype=np.int32)
    
        for i in range(nb_data):
            # Compute number of nearest neighbors per class
            knns_in_class = np.zeros((len(self.model._target_layers), self.nb_classes), dtype=np.int32)
    
            for layer_id, layer in enumerate(self.model._target_layers):
                knns_in_class[layer_id, :] = np.bincount(knns_labels[layer][i], minlength=self.nb_classes)
    
            # Compute number of knns in other class than class_id
            for class_id in range(self.nb_classes):
                knns_not_in_class[i, class_id] = np.sum(knns_in_class) - np.sum(knns_in_class[:, class_id])
    
        return knns_not_in_class
        
    def fprop(self, portion):
        """
        Performs a forward pass through the DkNN on an numpy array of data
        """
        print('---------- DkNN predict')
        print()
        if not self.calibrated:
            raise ValueError("DkNN needs to be calibrated by calling DkNNModel.calibrate method once before inferring")
            
        if portion == 'all':

            for ds_key in self.cv_dl:
                
                bs = self.cv_dl[ds_key].batch_size
                n_samples = len(self.cv_dl[ds_key].dataset)
                score = TensorDict(batch_size=n_samples)
                
                score['preds_knn'] = MMT.empty(shape=torch.Size((n_samples,)))
                score['confs'] = MMT.empty(shape=torch.Size((n_samples,)))
                score['creds'] = MMT.empty(shape=torch.Size((n_samples,)))
                score['p-value'] = MMT.empty(shape=torch.Size((n_samples,)+(self.nb_classes,)))
                # _dl = DataLoader(self.cv_dl[ds_key].dataset['in_activations'], batch_size=64, collate_fn = lambda x: x, shuffle=False)

                if self.verbose: print(f'\n ---- Getting scores for {ds_key}\n')
               
                # for i in range(n_samples):
                    
                knns_labels = {}
                         
                for layer in self.model._target_layers:
                    
                    filename = path.join(mkdtemp(), 'newfile.dat')
                    act = np.memmap(filename, dtype='float32', mode='w+', shape=self.cv_dl[ds_key].dataset['in_activations'][layer].flatten(start_dim=1).shape)
                    print(f'start finding_knns for {layer}')
                    act[:] = self.cv_dl[ds_key].dataset['in_activations'][layer].flatten(start_dim=1).detach().cpu().numpy()[:]
                    
                    _, knns_labels[layer] = self.find_train_knns(act, layer)
        
                # Calculate nonconformity
                knns_not_in_class = self.nonconformity(knns_labels)
        
                # Create predictions, confidence and credibility
                preds_knn, confs, creds, p_value = self.preds_conf_cred(knns_not_in_class)
                
                score['preds_knn'] = torch.Tensor(preds_knn)
                score['confs'] = torch.Tensor(confs)
                score['creds'] = torch.Tensor(creds)
                score['p-value'] = torch.Tensor(p_value)
                                    
                file_path = self.dknn_path/(ds_key)
                n_threads = 32
                if self.verbose: print(f'Saving {ds_key} to {file_path}.')
                score.memmap(file_path, num_threads=n_threads)
                self.res[ds_key] = score
        
        else:
            
            labels = self.cv_dl[portion].dataset['label']
        
            n_samples = len(labels)
            
            if self.percentage[portion] != 100:
        
                idx = np.arange(0, n_samples)
                rng = np.random.default_rng(self.seed)
                idx_rand = []
                
                for l in np.arange(0,self.nb_classes):
                
                    idx_l = np.argwhere(labels.numpy().astype(int)==l)            
                    n_samples_l = len(idx_l)
                    num_elements_to_select = int(n_samples_l*(self.percentage[portion]/100))            
                    rng.shuffle(idx_l)                
                    idx_rand.append(idx_l[:num_elements_to_select])            
                
                idx_rand = np.concatenate(idx_rand,axis=0)
            
                labels = labels[idx_rand[:,0]]
            
            knns_labels = {}
            for layer in self.model._target_layers:
                
                filename = path.join(mkdtemp(), 'newfile.dat')
                act = np.memmap(filename, dtype='float32', mode='w+', shape=self.cv_dl[portion].dataset['in_activations'][layer].flatten(start_dim=1).shape)
                act[:] = self.cv_dl[portion].dataset['in_activations'][layer].flatten(start_dim=1).detach().cpu().numpy()[:]
                
                if self.percentage[portion] != 100:
                    act = act[idx_rand[:,0]]

                _, knns_labels[layer] = self.find_train_knns(act, layer)
    
            #Calculate nonconformity
            knns_not_in_class = self.nonconformity(knns_labels)      
    
            # Create predictions, confidence and credibility
            preds_knn, confs, creds, p_value = self.preds_conf_cred(knns_not_in_class)
    
            return preds_knn, confs, creds, p_value 

        
    def preds_conf_cred(self, knns_not_in_class):
        """
        Given an array of nb_data x nb_classes dimensions, use conformal prediction to compute
        the DkNN's prediction, confidence and credibility
        """
        nb_data = knns_not_in_class.shape[0]
        preds_knn = np.zeros(nb_data, dtype=np.int32)
        confs = np.zeros(nb_data, dtype=np.float32)
        creds = np.zeros(nb_data, dtype=np.float32)
        p_value = np.zeros((nb_data, self.nb_classes), dtype=np.float32)

        for i in range(nb_data):
            # p-value of test input for each class
            # p_value = np.zeros(self.nb_classes, dtype=np.float32)

            for class_id in range(self.nb_classes):
                # p-value of (test point, candidate label)
                p_value[i][class_id] = (float(self.nb_cali) - bisect_left(self.cali_nonconformity, knns_not_in_class[i, class_id])) / float(self.nb_cali)

            preds_knn[i] = np.argmax(p_value[i])
            confs[i] = 1. - np.sort(p_value[i])[-2]
            creds[i] = np.max(p_value[i])

        return preds_knn, confs, creds, p_value
        