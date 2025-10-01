# torch stuff
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def feature_squeezing_score(**kwargs):

    '''
    Compute the Feature Squeezing score. Only computes it to samples which were successfully attacked. 

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - loaders (list[str]): List of loaders in `datasets.keys()` to compute corevectors. If `None` uses `datasets._dss.keys()`. Defaults to `None`.
    - Detector (peepholelib.featureSqueezing.FeatureSqueezingDetector): Detector used to analyse the input images
    - device (torch.device): device to perform the computations. 
    - batch_size (int): Creates dataloader to do computation in batch size. Defaults to 64.
    - n_threads (int): 'num_workers' passed to 'torch.utils.data.DataLoader'. Defaults to 1.
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - score_name (str): Name to use as key on the return dictionaty. Defaults to `Feature-Squeezing`. 
    - verbose (bool): print progress messages.

    Returns
    - ret (dict(str:dict(str:torch.tensor))): Scores as a two level dictionaty with the first key being the loaders, and second being the score name 'Proto-Class'. If 'append_scores' is passed, the dictionaries are appended.
    '''

    dss = kwargs.get('datasets')
    loaders = kwargs.get('loaders_ori', None)
    detector = kwargs.get('detector') 
    device = kwargs.get('device') 
    bs = kwargs.get('batch_size', 64)
    n_threads = kwargs.get('n_threads', 1) 
    append_scores = kwargs.get('append_scores', None)
    score_name = kwargs.get('score_name', 'Feature-Squeezing')
    verbose = kwargs.get('verbose', False)

    if loaders = None: loaders = datasets._dss.keys()
    if append_scores != None:
        ret = dict(append_scores)
    else:
        ret = {}

    for ds_key in loaders:
        if not ds_key in ret:
            ret[ds_key] = dict()

    for ds_key in loaders:
        n_samples = len(dss._dss[ds_key])
        dl = DataLoader(dss._dss[ds_key], batch_size=bs, shuffle=False, num_workers=n_threads)
        if verbose: print(f'Applying FS to {ds_key}')

        # TODO: this should be done in evaluation
        #idx = torch.argwhere((cv._dss[ds_key]['result']==1) & (cva._dss[ds_key]['attack_success']==1))
        
        # acc results
        ori_list = []
        for data in tqdm(dl):
            inputs = data['image'].to(device)
            output = detector(inputs)
            ori_list.append(output.detach().cpu())

        scores = torch.cat(ori_list, dim=0)
        ret[ds_key][score_name] = scores

    return ret
