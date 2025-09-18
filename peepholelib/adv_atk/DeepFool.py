# general python stuff
# torch stuff
import torchattacks

# our stuff
from .attack_base import AttackBase

class myDeepFool(AttackBase):
   
    def __init__(self, **kwargs):
        """
        'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
        [https://arxiv.org/abs/1511.04599]
        Distance Measure : L2
        Arguments:
            model (nn.Module): model to attack.
            steps (int): number of steps. (Default: 50)
            overshoot (float): parameter for enhancing the noise. (Default: 0.02)
        Shape:
            - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 <= y_i ` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        Examples::
            >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
            >>> adv_images = attack(images, labels)
        """
        AttackBase.__init__(self, **kwargs)
         
        self.steps = kwargs.get('steps', 50)
        self.overshoot = kwargs.get('overshoot', 0.02)

        self.atk = torchattacks.DeepFool(
                model=self.model._model,
                steps=self.steps,
                overshoot=self.overshoot
                )
        return

    def __call__(self, images, labels):
        return self.atk(images, labels)
