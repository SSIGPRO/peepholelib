# python stuff
import abc  
from pathlib import Path
from tqdm import tqdm
import numpy as np

# torch stuff
import torch
from peepholelib.peepholes.drill_base import DrillBase
from sklearn import covariance

# pur stuff
from peepholelib.models.model_wrap import get_out_activations

def get_images(**kwargs):
    """
    Get only images

    Args:
        act (PersistentTensorDict): TensorDict for the activations inside `peepholelib.CoreVectors` class

    Returns:
        image (torch.Tensor): images saved in the activations.  
    """
    img = kwargs['dss']['image']
    return img 

class DeepMahalanobisDistance(DrillBase): 
    def __init__(self, **kwargs):
        DrillBase.__init__(self, **kwargs)

        self.model = kwargs['model']
        self.magnitude = kwargs['magnitude']
        self.reducer = kwargs['reducer']
        self.ds_parser = kwargs.get('ds_parser', get_images)
        self.act_parser = kwargs.get('act_parser', get_out_activations)
        self.cv_parser = kwargs.get('cv_parser', self.reducer.parser)
        self.std_transform = torch.tensor(kwargs['std_transform'], device=self.device)
        self.save_input = kwargs.get('save_input', False)
        self.save_output = kwargs.get('save_output', True)

        # computed in fit()
        self._means = {} 
        self._precision = {}

        # set in fit()
        self._cvs = None 

        # used in save() or load()
        self.dmd_folder = self.path/self.name
        self.precision_path = self.dmd_folder/'precision.pt'
        self.mean_path = self.dmd_folder/'mean.pt' 

        # set model to save output activations
        # TODO: a bit ugly to leave this setted, should set both to False after finishing the peepholes computation
        self.model.set_activations(save_input=self.save_input, save_output=self.save_output)
        return
    
    def load(self, **kwargs):
        self._means = torch.load(self.mean_path).to(self.device)
        self._precision = torch.load(self.precision_path).to(self.device)
    
        return 

    def save(self, **kwargs):
        self.dmd_folder.mkdir(parents=True, exist_ok=True)
    
        torch.save(self._means.detach().cpu(), self.mean_path)
        torch.save(self._precision.detach().cpu(), self.precision_path)
        return     

    def fit(self, **kwargs):
        """
        compute sample mean and precision (inverse of covariance)
        return: sample_class_mean: list of class mean
                precision: list of precisions
        """

        _dss = kwargs['dataset']
        _cvs = kwargs['corevectors']
        loader = kwargs.get('loader')
        label_key = kwargs.get('label_key', 'label')
        
        # parsing for simplification
        dss = _dss._dss[loader]
        cvs = self.cv_parser(cvs=_cvs._corevds[loader][self.target_module])

        group_lasso = covariance.EmpiricalCovariance(assume_centered=False)
        
        # get TDs for each label
        labels = dss[label_key].int()
        self._means = torch.zeros(self.nl_model, self.n_features, device=self.device) 
        list_features = cvs.to(self.device) # create a copy of cvs to device
        
        for i in range(self.nl_model):
            self._means[i] = list_features[labels == i].mean(dim=0).to(self.device)
            list_features[labels == i] -= self._means[i]
        
        # find inverse            
        group_lasso.fit(list_features.cpu().numpy())
        self._precision = torch.from_numpy(group_lasso.precision_).float().to(self.device)
        return 

    def __call__(self, **kwargs):
        '''
        Compute the proposed Mahalanobis confidence score on input dataset
        return: Mahalanobis score from layer_index It computes the score for a single layer at the time

        sample_mean is a DICT of a number of elemnts that is equal to the number of layers present in target layers. Each element is a tensor 
        of dim (n_classes, dim of the coreavg) in layer features.28 it will be(100,512)
        precision is a DICT of precision matrices one for each layer
        '''

        magnitude = self.magnitude
        std = self.std_transform
        
        dss = kwargs['dss']

        # get input image and set gradient to modify it
        data = self.ds_parser(dss = dss)
        data = data.to(self.device)
        data.requires_grad_(True)
        n_samples = data.shape[0]

        self.model._model.eval()

        self.model._model.zero_grad()
        _ = self.model(data.to(self.device))
        
        if self.target_module == 'output':
            output = self.model(data.to(self.device))
        else:
            _parsed_act = self.act_parser(self.model._acts)[self.target_module]
            output = self.cv_parser(cvs=self.reducer(act_data=_parsed_act))
        
        gaussian_score = torch.zeros(n_samples, self.nl_model, device=self.device)
        for i in range(self.nl_model):
            zero_f = output - self._means[i]
            term_gau = -0.5*torch.mm(torch.mm(zero_f, self._precision), zero_f.t()).diag()
            gaussian_score[:,i] = term_gau

        if magnitude != 0:
            # Input_processing
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = self._means[sample_pred]
            zero_f = output - batch_sample_mean
            pure_gau = -0.5*torch.mm(torch.mm(zero_f, self._precision), zero_f.t()).diag()
            loss = torch.mean(-pure_gau)
            loss.backward()
            
            gradient = torch.ge(data.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            for i in range(3):
                gradient.index_copy_(1, torch.LongTensor([i]).to(self.device), gradient.index_select(1, torch.LongTensor([i]).to(self.device)) / (std[i]))
        
            tempInputs = torch.add(data.data, gradient, alpha=-magnitude)

            with torch.no_grad():
                _ = self.model(tempInputs.to(self.device))
                
            if self.target_module == 'output':
                output = self.model(tempInputs.to(self.device)) 
            else:
                _parsed_act = self.act_parser(self.model._acts)[self.target_module]
                output = self.cv_parser(cvs=self.reducer(act_data=_parsed_act))

            gaussian_score = torch.zeros(n_samples, self.nl_model, device=self.device)
            for i in range(self.nl_model):
                zero_f = output - self._means[i]
                term_gau = -0.5*torch.mm(torch.mm(zero_f, self._precision), zero_f.t()).diag()
                gaussian_score[:, i] = term_gau

        score = gaussian_score
        return score.detach().cpu()
