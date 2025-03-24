# python stuff
import abc  
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# torch stuff
import torch
from peepholelib.peepholes.drill_base import DrillBase
from sklearn import covariance

class DeepMahalanobisDistance(DrillBase): 
    def __init__(self, **kwargs):
        DrillBase.__init__(self, **kwargs)

        self.model = kwargs['model']
        self._layer = kwargs['layer']
        self.magnitude = kwargs['magnitude']
        self.std_transform = torch.tensor(kwargs['std_transform'], device=self.device)

        # used in __call__()
        self.hooks = self.model.get_hooks()

        # computed in fit()
        self._means = {} 
        self._precision = {}

        # set in fit()
        self._cvs = None 

        # defined in save() or load()
        self._mean_file = None
        self._precision_file = None
        self._suffix = f'{self.name}.nl_model={self.nl_model}'
        self.dmd_folder = None
        return
    
    def load(self, **kwargs):
        if self.dmd_folder == None:
            self.dmd_folder = self.path/self._suffix

        self.precision_path = self.dmd_folder/'precision.pt'
        self.mean_path = self.dmd_folder/'mean.pt' 

        self._means = torch.load(self.mean_path).to(self.device)
        self._precision = torch.load(self.precision_path).to(self.device)
    
        return 

    def save(self, **kwargs):
        if self.dmd_folder == None:
            self.dmd_folder = self.path/self._suffix
        
        self.dmd_folder.mkdir(parents=True, exist_ok=True)

        self.precision_path = self.dmd_folder/'precision.pt'
        self.mean_path = self.dmd_folder/'mean.pt'   
    
        torch.save(self._means.detach().cpu(), self.mean_path)
        torch.save(self._precision.detach().cpu(), self.precision_path)
        return     

    def fit(self, **kwargs):
        """
        compute sample mean and precision (inverse of covariance)
        return: sample_class_mean: list of class mean
                precision: list of precisions
        """

        cvs = kwargs['corevectors']
        acts = kwargs['activations']
        label_key = kwargs['label_key'] if 'label_key' in kwargs else 'label'
        
        group_lasso = covariance.EmpiricalCovariance(assume_centered=False)
        
        # get TDs for each label
        labels = acts[label_key].int()
        self._means = torch.zeros(self.nl_model, self.n_features, device=self.device) 
        list_features = cvs.clone().detach().to(self.device) # create a copy of cvs to device
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
        
        acts = kwargs['acts']
        data = self.parser(act = acts, **self.parser_kwargs)
        data = data.to(self.device)
        data.requires_grad_(True)
        n_samples = data.shape[0]

        # compute Mahalanobis score
        gaussian_score = 0
        self.model._model.eval()

        # TODO: need this reset?
        #self.hooks[self._layer].in_activations = None 
        _ = self.model(data.to(self.device))
        
        output = self.hooks[self._layer].in_activations[:]
        output = output.view(output.size(0), output.size(1), -1)
        output = torch.mean(output, 2)
        output = output.to(self.device)
        
        gaussian_score = torch.zeros(n_samples, self.nl_model)
        for i in range(self.nl_model):
            zero_f = output - self._means[i]
            term_gau = -0.5*torch.mm(torch.mm(zero_f, self._precision), zero_f.t()).diag()
            gaussian_score[:,i] = term_gau

        # Input_processing
        sample_pred = gaussian_score.max(1)[1].to(self.device)
        batch_sample_mean = self._means.index_select(0, sample_pred)
        zero_f = output - batch_sample_mean
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, self._precision), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
        
        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # TODO: Still think this could be simples
        for i in range(3):
            gradient.index_copy_(1, torch.LongTensor([i]).to(self.device), gradient.index_select(1, torch.LongTensor([i]).to(self.device)) / (std[i]))
    
        tempInputs = torch.add(data.data, -magnitude, gradient)

        # TODO: is this necessary?
        #self.hooks[self._layer].in_activations = None 
        with torch.no_grad():
            _ = self.model(tempInputs.to(self.device))
        output = self.hooks[self._layer].in_activations[:]
        output = output.view(output.size(0), output.size(1), -1)
        output = torch.mean(output, 2)
        output = output.to(self.device)
        noise_gaussian_score = 0

        noise_gaussian_score = torch.zeros(n_samples, self.nl_model, device=self.device)
        for i in range(self.nl_model):
            zero_f = output - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, self._precision), zero_f.t()).diag()
            noise_gaussian_score[:, i] = term_gau
        
        # TODO: ref code returns noise_gaussian_score
        input('.............. wait................. ')
        return noise_gaussian_score.detach().cpu()
