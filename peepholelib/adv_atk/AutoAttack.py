# torch stuff
from autoattack import AutoAttack

# our stuff
from .attack_base import AttackBase

class myAutoAttack(AttackBase):
   
    def __init__(self, **kwargs):
        """
        AutoAttack in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
        [https://arxiv.org/abs/2003.01690]
    
        Distance Measure : Linf
    
        Arguments:
            model (nn.Module): model to attack.
            eps (float): maximum perturbation. (Default: 8/255)
            version (str): 'standard', 'plus' or 'rand'. (Default: 'standard')
            attack_to_run (str): 'apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t'. (Default: 'apgd-ce')
    
        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 >= y_i >=` `number of labels`.
            - output: :math:`(N, C, H, W)`.
    
        Examples::

            >>> adversary = AutoAttack(
                    model,
                    norm='Linf',
                    eps=8/255,
                    version='standard'  # same as RobustBench
                )

            >>> x_adv = adversary.run_standard_evaluation(
                    x_test,
                    y_test,
                    bs=128
                )
    
        """
        AttackBase.__init__(self, **kwargs)
         
        self.eps = kwargs.get('eps', 8/255)
        self.norm = kwargs.get('norm', 'Linf')
        self.version = kwargs.get('version', 'standard')
        self.attack_to_run = kwargs.get('attack_to_run', 'apgd-ce')

        adversary = AutoAttack(
                        model=self.model._model,
                        norm=self.norm,
                        eps=self.eps,
                        version=self.version,
                        device=self.model.device
                    )

        if self.version == 'standard':
            valid_attacks = ['apgd-ce', 'apgd-t', 'fab-t', 'square']

            if self.norm in ['Linf', 'L2']:
                adversary.apgd.n_restarts = 1
                adversary.apgd_targeted.n_target_classes = 9
            elif self.norm in ['L1']:
                adversary.apgd.use_largereps = True
                adversary.apgd_targeted.use_largereps = True
                adversary.apgd.n_restarts = 5
                adversary.apgd_targeted.n_target_classes = 5
            adversary.fab.n_restarts = 1
            adversary.apgd_targeted.n_restarts = 1
            adversary.fab.n_target_classes = 9
            adversary.square.n_queries = 5000

        elif self.version == 'plus':
            valid_attacks = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self.apgd.n_restarts = 5
            self.fab.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
            if not self.norm in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(self.version, self.norm))

        elif self.version == 'rand':
            valid_attacks = ['apgd-ce', 'apgd-dlr']
            adversary.apgd.n_restarts = 1
            adversary.apgd.eot_iter = 20
        
        else: 
            raise ValueError(f'Invalid version: {self.version} choose among <standard|plus|rand>')

        if self.attack_to_run not in valid_attacks:
            raise ValueError(f'Invalid attack: {self.attack_to_run}')

        adversary.attacks_to_run = [self.attack_to_run]
        self.atk = adversary
        return            
      
    def __call__(self, images, labels):
        return self.atk.run_standard_evaluation(images, labels)
