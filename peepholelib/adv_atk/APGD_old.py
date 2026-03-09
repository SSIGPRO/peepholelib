# torch stuff
import torchattacks

# our stuff
from .attack_base import AttackBase

class myAPGD(AttackBase):
    def __init__(self, **kwargs):
        """
        APGD in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse 
        parameter-free attacks' [https://arxiv.org/abs/2003.01690] [https://github.com/fra31/auto-attack]
    
        Distance Measure : Linf, L2
    
        Arguments:
            model (nn.Module) : model to attack.
            norm (str) : Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
            eps (float) : maximum perturbation. (Default: 8/255)
            steps (int) : number of steps. (Default: 10)
            n_restarts (int) : number of random restarts. (Default: 1)
            seed (int) : random seed for the starting point. (Default: 0)
            loss (str) : loss function optimized. ['ce', 'dlr'] (Default: 'ce') where CE is cross entropy and DLR is Difference of Logits Ratio Loss
            eot_iter (int) : number of iteration for EOT. (Default: 1)
            rho (float) : parameter for step-size update (Default: 0.75)
            verbose (bool) : print progress. (Default: False)
            n_classes (int) : number of classes. (Default: 10), this is for targeted version, note that it drops loss as an argument
        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,`H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 <= y_i <= `number of labels`.
            - output: :math:`(N, C, H, W)`.
    
        Examples::
            attack = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
            adv_images = attack(images, labels)

            for targeted:
            attack = torchattacks.APGDT(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, eot_iter=1, rho=.75, verbose=False, n_classes=10)
            adv_images = attack(images, labels)
        """

        AttackBase.__init__(self, **kwargs)

        self.norm = kwargs.get('norm','Linf')
        self.eps = kwargs.get('eps', 0.5/255)
        self.steps = kwargs.get('steps', 10)
        self.n_restarts = kwargs.get('n_restarts', 1)
        self.seed = kwargs.get('seed', 0)
        self.loss= kwargs.get('loss','dlr')
        self.eot_iter = kwargs.get('eot_iter', 1)
        self.rho = kwargs.get('rho', 0.75)
        self.verbose = kwargs.get('verbose', False)
        self.n_classes = kwargs.get('n_classes', 100)
        self.mode = kwargs.get('mode','random')
        
        self.AP = kwargs.get('AP', 'APGDT')

        if self.AP == 'APGD':
                self.atk = torchattacks.APGD(
                model = self.model._model,
                norm=self.norm,
                eps = self.eps,  
                steps = self.steps,
                n_restarts = self.n_restarts,
                seed = self.seed,
                loss = self.loss,
                eot_iter = self.eot_iter,
                rho = self.rho,
                verbose = self.verbose
                )

        elif self.AP == 'APGDT':
                self.atk = torchattacks.APGDT(
                model = self.model._model, 
                norm=self.norm,
                eps = self.eps,  
                steps = self.steps,
                n_restarts = self.n_restarts,
                seed = self.seed,
                eot_iter = self.eot_iter,
                rho = self.rho,
                verbose = self.verbose,
                n_classes = self.n_classes
                )

                # # following not supported since they default APGDT as non targeted while it is targeted (wtf?)
                # if self.mode == 'random':
                #     self.atk.set_mode_targeted_random(quiet=False)
                # elif self.mode == 'least-likely':
                #     self.atk.set_mode_targeted_least_likely(kth_min=1, quiet=False)
    @torch.no_grad()
    def least_likely_targets(model, images, true_labels):
        logits = model(images)
        logits = logits.clone()
        logits.scatter_(1, true_labels.view(-1,1), float("inf"))  # exclude true
        return torch.argmin(logits, dim=1)

    def __call__(self, images, labels):
        return self.atk(images, labels)