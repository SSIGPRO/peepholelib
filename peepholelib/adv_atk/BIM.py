import torchattacks

import torch
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

from pathlib import Path as Path
import abc 

from .attacks_base import AttackBase
from tqdm import tqdm


class myBIM(AttackBase):
   
    def __init__(self, **kwargs):
        AttackBase.__init__(self, **kwargs)
        """
        BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
        [https://arxiv.org/abs/1607.02533]
    
        Distance Measure : Linf
    
        Arguments:
            model (nn.Module): model to attack.
            eps (float): maximum perturbation. (Default: 8/255)
            alpha (float): step size. (Default: 2/255)
            steps (int): number of steps. (Default: 10)
    
        .. note:: If steps set to 0, steps will be automatically decided following the paper.
    
        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
    
        Examples::
            >>> attack = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
            >>> adv_images = attack(images, labels)
        """
        print('---------- Attack BIM init')
        print()
         
        self._loaders = kwargs['dl']
        self.model = kwargs['model']
        self.name_model = kwargs['name_model']
        self.eps = kwargs['eps'] if 'eps' in kwargs else 8/255
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 2/255
        self.steps = kwargs['steps'] if 'steps' in kwargs else 10
        self.verbose = kwargs['verbose'] if 'verbose' in kwargs else True
        self.device = kwargs['device'] 
        self.atk_path = self.path/Path(f'model_{self.name_model}/eps_{self.eps:.2f}/alpha_{self.alpha:.2f}/steps_{self.steps}')
        self.mode = kwargs['mode'] if 'mode' in kwargs else 'random'


        if self.atk_path.exists():
            self._atkds = {}
            if self.verbose: print(f'File {self.atk_path} exists.')
            for ds_key in self._loaders:
                self._atkds[ds_key] = TensorDict.load_memmap(self.atk_path/ds_key)
        else:
            
            self.atk = torchattacks.BIM(model=self.model, 
                                        eps=self.eps, 
                                        alpha=self.alpha, 
                                        steps=self.steps)

            if self.mode == 'random':
                self.atk.set_mode_targeted_random(quiet=False)
            elif self.mode == 'least-likely':
                self.atk.set_mode_targeted_least_likely(kth_min=1, quiet=False)
                self.atk.get_least_likely_label

    def get_ds_attack(self):
        self.atk_path.mkdir(parents=True, exist_ok=True)

        attack_TensorDict = {}
        
        for loader_name in self._loaders:
            
            if self.verbose: print(f'\n ---- Getting data from {loader_name}\n')
            n_samples = len(self._loaders[loader_name].dataset)

            if self.verbose: print('loader n_samples: ', n_samples) 
            #TODO: check device
            attack_TensorDict[loader_name] = TensorDict(batch_size=n_samples) 

            file_path = self.atk_path/(loader_name)
            n_threads = 32
            
            bs = self._loaders[loader_name].batch_size
            _img, _ = self._loaders[loader_name].dataset[0]
            
            attack_TensorDict[loader_name]['image'] = MMT.empty(shape=torch.Size((n_samples,)+_img.shape))
            attack_TensorDict[loader_name]['label'] = MMT.empty(shape=torch.Size((n_samples,)))
            attack_TensorDict[loader_name]['attack_success'] = MMT.empty(shape=torch.Size((n_samples,)))
            for bn, data in enumerate(tqdm(self._loaders[loader_name])):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                n_in = len(images)
                attack_images = self.atk(images, labels)
                
                with torch.no_grad():
                        y_predicted = self.model(attack_images)
                predicted_labels = y_predicted.argmax(axis = 1)
                results = predicted_labels != labels

                attack_TensorDict[loader_name]['image'][bn*bs:bn*bs+n_in] = images
                attack_TensorDict[loader_name]['label'][bn*bs:bn*bs+n_in] = labels
                attack_TensorDict[loader_name]['attack_success'][bn*bs:bn*bs+n_in] = results                
            
            # if self.verbose: print(f'Saving {loader_name} to {file_path}.')
            attack_TensorDict[loader_name].memmap(file_path, num_threads=n_threads)
            self._atkds = attack_TensorDict

