# torch stuff
import torch
from torch.utils.data import DataLoader

def DOCTOR_score(**kwargs):
    '''
    Compute DOCTOR score described in https://arxiv.org/pdf/2106.02395

    Args:
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed datasets.
    - loaders (list[str]): loaders to consider, usually `['train', 'test', 'val']`, if `None`, gets all loaders in `datasets._dss`. Defaults to `None`.
    - net (torch.nn.Module): model used to compute the logits.
    - temperature (float): temperature factor. Defaults to 1.0.
    - magnitude (float): magnitude of the adversarial perturbation.
    - append_scores (dict): Append the scores form this dictionaty to the scores computed in this function. Overwrite if same keys.
    - verbose (bool): print progress messages.
    - bs (int): batch size used to compute the scores.
    - device (torch.device): device where to perform the computations.
    - n_threads (int): number of workers used in the dataloader. Defaults to 1.
    '''

    dss = kwargsa.get('datasets')
    loaders = kwargs.get('loaders', None)
    temperature = kwargs.get('temperature', 1.)
    magnitude = kwargs.get('magnitude', 0.)
    append_scores = kwargs.get('append_scores', None)
    verbose = kwargs.get('verbose', False)
    model = kwargs.get('net', None)
    device = kwargs.get('device')
    n_threads = kwargs.get('n_threads', 32)
    bs = kwargs.get('bs', 128)

    # parse arguments
    if loaders == None: loaders = list(dss._dss.keys())
    score_name = 'DOCTOR'

    # create the return dictionary. 
    if append_scores != None:
        ret = dict(append_scores)
    else:
        ret = {}
    
    for ds_key in loaders:
        if not ds_key in ret:
            ret[ds_key] = dict()

    net = model._model
    net.to(device)
    net.eval()

    for ds_key in loaders:
        dssds = dss._dss[ds_key]

        n_samples = len(dssds)
        
        ret[ds_key][score_name] = torch.empty(n_samples, dtype=torch.float32)
        
        dl_dss = DataLoader(dssds, batch_size=bs, collate_fn=lambda x: x, num_workers = n_threads, shuffle=False)
        
        write_ptr = 0
    
        for _dss in tqdm(dl_dss,total=ceil(n_samples/bs)):

            inputs = _dss['image'].to(device)
            
            if magnitude == 0:
                output = _dss['output'].to(device)
            else:
                inputs.requires_grad_(True)
                model._model.zero_grad()

                output = net(inputs)
                scores = torch.sum(sm(output/temperature, dim=1)**2, dim=1)
            
                log_scores = torch.log(torch.clamp(scores, min=1e-12))
                log_scores.sum().backward()

                new_inputs = inputs + magnitude*torch.sign(inputs.grad)
                new_inputs = new_inputs.clamp(0, 1).detach()

                inputs.requires_grad_(False)
                net.zero_grad(set_to_none=True)
                with torch.no_grad():

                    output = net(new_inputs)

            scores = torch.sum(sm(output/temperature, dim=1)**2, dim=1)
            bsz = scores.shape[0]

            ret[ds_key][score_name][write_ptr:write_ptr+bsz] = scores.detach().cpu()

            write_ptr += bsz
                
    return ret       

