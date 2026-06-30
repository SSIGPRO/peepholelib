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
            attacks_to_run (str): 'apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t'. (Default: None, runs all attacks for the chosen version)
    
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
         
        eps = kwargs.get('eps', 8/255)
        norm = kwargs.get('norm', 'Linf')
        version = kwargs.get('version', 'standard')
        self.attacks_to_run = kwargs.get('attacks_to_run', None)
        fab_n_restarts = kwargs.get('fab_n_restarts', 5)
        fab_n_target_classes = kwargs.get('fab_n_target_classes', 20)
        square_n_queries = kwargs.get('square_n_queries', 20000)

        adversary = AutoAttack(
                model = self.model,
                norm = norm,
                eps = eps,
                version = version,
                device = self.model.device
                )

        if version == 'standard':
            valid_attacks = ['apgd-ce', 'apgd-t', 'fab-t', 'square']

            if norm in ['Linf', 'L2']:
                adversary.apgd.n_restarts = 1
                adversary.apgd_targeted.n_target_classes = 9
            elif norm in ['L1']:
                adversary.apgd.use_largereps = True
                adversary.apgd_targeted.use_largereps = True
                adversary.apgd.n_restarts = 5
                adversary.apgd_targeted.n_target_classes = 5
            adversary.fab.n_restarts = fab_n_restarts
            adversary.fab.n_target_classes = fab_n_target_classes
            adversary.square.n_queries = square_n_queries
            adversary.apgd_targeted.n_restarts = 1
            

        elif version == 'plus':
            valid_attacks = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            adversary.apgd.n_restarts = 5
            adversary.fab.n_restarts = 5
            adversary.apgd_targeted.n_restarts = 1
            adversary.fab.n_target_classes = 9
            adversary.apgd_targeted.n_target_classes = 9
            adversary.square.n_queries = 5000
            if not norm in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(version, norm))

        elif version == 'rand':
            valid_attacks = ['apgd-ce', 'apgd-dlr']
            adversary.apgd.n_restarts = 1
            adversary.apgd.eot_iter = 20
        
        else: 
            raise ValueError(f'Invalid version: {version} choose among <standard|plus|rand>')

        if self.attacks_to_run is not None:
            if len(set(self.attacks_to_run)-set(valid_attacks)) > 0:
                raise ValueError(f'Invalid attack: {self.attacks_to_run}')
            adversary.attacks_to_run = self.attacks_to_run
        self.atk = adversary
        return            
      
    def __call__(self, images, labels):
        return self.atk.run_standard_evaluation(images, labels)
