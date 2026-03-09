# torch stuff
import torchattacks

# our stuff
from .attack_base import AttackBase

class BIMStoreTarget(torchattacks.BIM):
    def get_target_label(self, inputs, labels=None):
        y_t = super().get_target_label(inputs, labels)
        self.y_target = y_t.detach().clone()
        return y_t

class myBIM(AttackBase):
    def __init__(self, **kwargs):
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
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 <= y_i <= `number of labels`.
            - output: :math:`(N, C, H, W)`.
    
        Examples::
            attack = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
            adv_images = attack(images, labels)
        """

        AttackBase.__init__(self, **kwargs)

        self.eps = kwargs.get('eps', 4/255)
        self.alpha = kwargs.get('alpha', 0.05/255)
        self.steps = kwargs.get('steps', 100)
        self.mode = kwargs.get('mode', 'random')

        self.atk = BIMStoreTarget(
                model = self.model._model, 
                eps = self.eps, 
                alpha = self.alpha, 
                steps = self.steps
                )

        if self.mode == 'random':
            self.atk.set_mode_targeted_random(quiet=False)
        elif self.mode == 'least-likely':
            self.atk.set_mode_targeted_least_likely(kth_min=1, quiet=False)
        
        # return removed it as is probably not needed
            # self.atk.get_least_likely_label # probably not needed
        # atk.set_mode_targeted_by_label()
        # Suppose CIFAR-10 class index for "plane" is 0
        # plane_label = 0
        # target_labels = torch.full_like(labels, plane_label)
        # adv_images = atk(images, target_labels)
    def __call__(self, images, labels):
        return self.atk(images, labels)
