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
from peepholelib.coreVectors.dimReduction.avgPooling import ChannelWiseMean_conv
from sklearn import covariance

class DeepMahalanobisDistance(DrillBase): 
    def __init__(self, **kwargs):
        DrillBase.__init__(self, **kwargs)

        self.model = kwargs['model']
        self._layer = kwargs['layer']
        self.magnitude = kwargs['magnitude']
        self.googlstd_transform = torch.tensor(kwargs['std_transform'], device=self.device)
        self.parser_act = kwargs['parser_act'] if 'parser_act' in kwargs else ChannelWiseMean_conv
        self.save_input = kwargs['save_input'] if 'save_input' in kwargs else False
        self.save_output = kwargs['save_output'] if 'save_output' in kwargs else True

        # computed in fit()
        self._means = {} 
        self._precision = {}

        # set in fit()
        self._cvs = None 

        # used in save() or load()
        self.dmd_folder = self.path/(self.name+self._suffix)
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
        drill_key = kwargs.get('drill_key')
        label_key = kwargs.get('label_key', 'label')
        
        # parsing for simplification
        dss = _dss._dss[loader]
        cvs = _cvs._corevds[loader][drill_key]
        #TODO i had to make cvs with dimension 512 in order to perform this computation
        # check if the output isoutput layerwise oroutput of model
        #print(f'cvs.shape{cvs.shape}')# ->[500,100] before
        #quit()

        group_lasso = covariance.EmpiricalCovariance(assume_centered=False)
        
        # get TDs for each label
        labels = dss[label_key].int()
        #self._means = torch.zeros(self.nl_model, self.n_features, device=self.device) 
        list_features = cvs.clone().detach().to(self.device) # create a copy of cvs to device
        #Elma-> 
        self._means = torch.zeros(self.nl_model, list_features.shape[1], device=self.device) 
        
        #print(f'self.means{self._means.shape}\nself.nl_model{self.nl_model}\nlist_features{list_features.shape}')
        #quit()
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
        parser_act = kwargs['parser_act'] if 'parser_act' in kwargs else ChannelWiseMean_conv

        # get input image and set gradient to modify it
        data = self.parser(dss = dss)
        data = data.to(self.device)
        data.requires_grad_(True)
        n_samples = data.shape[0]

        self.model._model.eval()

        self.model._model.zero_grad()
        _ = self.model(data.to(self.device))
        
        if self._layer == 'output':
            output = self.model(data.to(self.device))
        else:
            output = self.parser_act(self.model._acts['out_activations'][self._layer])
        
        gaussian_score = torch.zeros(n_samples, self.nl_model, device=self.device)
        #print(f'output.shape{output.shape}\nself.means.shape{self._means.shape}')
        #quit()
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

            # TODO: Still think this could be simpler
            # TODO: is this 3 because the activation are reshaped to have 3 dimensions?
            # I suspect this is specific for the models used in the reference code
            # The std values should probably reflect the number of dimensions
            for i in range(3):
                #print(f'gradient.shape{gradient.shape}')
                #print(f'i{i}')
                #print(f'std.shape{std.shape}')
                #print(f'std[i]{std[i]}')
                #quit()
                #------Breaking Hete------------
                gradient.index_copy_(1, torch.LongTensor([i]).to(self.device), gradient.index_select(1, torch.LongTensor([i]).to(self.device)) / (std[i]))
                #do it 
                '''
                idx = torch.tensor([i], dtype = torch.long, device = self.device)#.to(self.device)
                slice_ = gradient.index_select(1,idx)
                slice_ = slice_.squeeze(1)
                slice_i = slice_ / std[i]
                gradient.index_copy_(1, idx,slice_i.unsqueeze(1))
                '''
            tempInputs = torch.add(data.data, gradient, alpha=-magnitude)

            with torch.no_grad():
                _ = self.model(tempInputs.to(self.device))
                
            if self._layer == 'output':
                output = self.model(tempInputs.to(self.device)) 
            else:
                output = self.parser_act(self.model._acts['out_activations'][self._layer])

            gaussian_score = torch.zeros(n_samples, self.nl_model, device=self.device)
            for i in range(self.nl_model):
                zero_f = output - self._means[i]
                term_gau = -0.5*torch.mm(torch.mm(zero_f, self._precision), zero_f.t()).diag()
                gaussian_score[:, i] = term_gau

        score = gaussian_score
        return score.detach().cpu()
