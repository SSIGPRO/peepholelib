# torch stuff
import torchattacks

# our stuff
from .attack_base import AttackBase

class myAPGD(AttackBase):
   
    def __init__(self, **kwargs):
        """
        PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
        [https://arxiv.org/abs/1706.06083]
    
        Distance Measure : Linf
    
        Arguments:
            model (nn.Module): model to attack.
            eps (float): maximum perturbation. (Default: 8/255)
            alpha (float): step size. (Default: 2/255)
            steps (int): number of steps. (Default: 10)
            random_start (bool): using random initialization of delta. (Default: True)
    
        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 >= y_i >=` `number of labels`.
            - output: :math:`(N, C, H, W)`.
    
        Examples::
            >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
            >>> adv_images = attack(images, labels)
    
        """
        AttackBase.__init__(self, **kwargs)
         
        self.eps = kwargs.get('eps', 8/255)
        self.steps = kwargs.get('steps', 10)
        self.n_restarts = kwargs.get('n_restarts', 1)
        self.loss = kwargs.get('loss', 'ce')
        self.eot_iter = kwargs.get('eot_iter', 1)
        self.rho = kwargs.get('rho', 0.75)
        self.mode = kwargs.get('mode', 'random')
        
        self.atk = torchattacks.APGD(
                model=self.model._model, 
                eps=self.eps, 
                steps=self.steps,
                n_restarts=self.n_restarts,
                loss=self.loss,
                eot_iter=self.eot_iter,
                rho=self.rho,
                )

        if self.mode == 'random':
            self.atk.set_mode_targeted_random(quiet=False)
        elif self.mode == 'least-likely':
            self.atk.set_mode_targeted_least_likely(kth_min=1, quiet=False)
            self.atk.get_least_likely_label
        return            
      
    def __call__(self, images, labels):
        return self.atk(images, labels)
