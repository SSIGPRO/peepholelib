# python stuff
import abc  
from pathlib import Path
from tqdm import tqdm
import numpy as np

# torch stuff
import torch
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader
from peepholelib.Drillers.drill_base import DrillBase
from sklearn import covariance

def null_parser(**kwargs):
    data = kwargs['data']
    return data['data'], data['label'] 
    
class DeepMahalanobisDistance(DrillBase): 
    def __init__(self, **kwargs):
        DrillBase.__init__(self, **kwargs)

        self.model = kwargs['model']
        self.magnitude = kwargs['magnitude']
        self.mean_transform = kwargs['mean_transform']

        # computed in fit()
        self._mean = {} 
        self._precision = {}

        # set in fit()
        self._cvs = None 

        # defined in save() or load()
        self._clas_file = None
        self._suffix = f'{self.name}.nl_model={self.nl_model}'
        return
    
    def load(self, **kwargs):
        pass 

    def save(self, **kwargs):
        pass       

    def fit(self, **kwargs):
        """
        compute sample mean and precision (inverse of covariance)
        return: sample_class_mean: list of class mean
                precision: list of precisions
        """
        print(self.name)
        cvs = kwargs['corevectors'][self.name]
        acts = kwargs['activations']
        
        group_lasso = covariance.EmpiricalCovariance(assume_centered=False)
        
        num_sample_per_class = np.zeros(self.nl_model)
        list_features = []
        for j in range(self.nl_model):
            list_features.append(0)
        ## it creats a list of n_classes
        
        for i, (act, cv) in tqdm(enumerate(zip(acts, cvs))):
                
            label = int(act['label']) 
            
            if num_sample_per_class[label] == 0:
                list_features[label] = cv.view(1, -1)
            else:
                 list_features[label] = torch.cat((list_features[label], cv.view(1, -1)), 0)

            num_sample_per_class[label] += 1
        print(cvs.size())
        print(cvs[0].size())        
        _size = cvs.size(1)
        mean_list = torch.Tensor(self.nl_model, _size).cuda()
        for j in range(self.nl_model):
            mean_list[j] = torch.mean(list_features[j], 0)
        self._mean = mean_list
        print(self._mean)
        
        X = 0
        for i in range(self.nl_model):
            if i == 0:
                X = list_features[i].to(self.device) - self._mean[i].to(self.device)
                print(X)

            else:
                X = torch.cat((X, list_features[i].to(self.device) - self._mean[i].to(self.device)), 0)       
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        precision = group_lasso.precision_
        precision = torch.from_numpy(precision).float().cuda()
        self._precision = precision

        return 

    def classifier_probabilities(self, **kwargs):
        pass 
    def __call__(self, **kwargs):
        '''
        Compute the proposed Mahalanobis confidence score on input dataset
        return: Mahalanobis score from layer_index It computes the score for a single layer at the time


        sample_mean is a DICT of a number of elemnts that is equal to the number of layers present in target layers. Each element is a tensor 
        of dim (n_classes, dim of the coreavg) in layer features.28 it will be(100,512)
        precision is a DICT of precision matrices one for each layer
        '''

        acts = kwargs['acts']
        model = self.model
        magnitude = self.magnitude
        mean = self.mean_transform

        data = torch.tensor(acts['image'], requires_grad=True, device=self.device)

        hooks = model.get_hooks()

        # compute Mahalanobis score
        gaussian_score = 0
        model._model.eval()
        hooks[self.name].in_activations = None 
        _ = model(data.to(self.device))
        
        output = hooks[self.name].in_activations[:]
        output = output.view(output.size(0), output.size(1), -1)
        output = torch.mean(output, 2)
        output = output.to(self.device)

        self._precision = self._precision.to(self.device)
        for i in range(self.nl_model):
            batch_sample_mean = self._mean[i].to(self.device)
            
            zero_f = output - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, self._precision), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        # Input_processing
        sample_pred = gaussian_score.max(1)[1].to(self.device)
        batch_sample_mean = self._mean.to(self.device).index_select(0, sample_pred)
        zero_f = output - batch_sample_mean
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, self._precision), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
        
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        gradient.index_copy_(1, torch.LongTensor([0]).to(self.device), gradient.index_select(1, torch.LongTensor([0]).to(self.device)) / (mean[0]))
        gradient.index_copy_(1, torch.LongTensor([1]).to(self.device), gradient.index_select(1, torch.LongTensor([1]).to(self.device)) / (mean[1]))
        gradient.index_copy_(1, torch.LongTensor([2]).to(self.device), gradient.index_select(1, torch.LongTensor([2]).to(self.device)) / (mean[2]))
    
        tempInputs = torch.add(data.data, -magnitude, gradient)
        hooks[self.name].in_activations = None 
        with torch.no_grad():
            _ = model(tempInputs.to(self.device))
        output = hooks[self.name].in_activations[:]
        output = output.view(output.size(0), output.size(1), -1)
        output = torch.mean(output, 2)
        output = output.to(self.device)
        noise_gaussian_score = 0
        for i in range(self.nl_model):
            batch_sample_mean = self._mean[i].to(self.device)
            
            zero_f = output - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, self._precision), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)   

        return gaussian_score.detach()
