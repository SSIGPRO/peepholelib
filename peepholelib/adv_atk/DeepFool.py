# general python stuff
from pathlib import Path as Path

# torch stuff
import torchattacks

# our stuff
from .attacks_base import AttackBase

class myDeepFool(AttackBase):
   
    def __init__(self, **kwargs):
        AttackBase.__init__(self, **kwargs)
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
            - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
            - output: :math:`(N, C, H, W)`.
        Examples::
            >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
            >>> adv_images = attack(images, labels)
        """
        print('---------- Attack DeepFool init')
         
        self.steps = kwargs.get('steps', 50)
        self.overshoot = kwargs.get('overshoot', 0.02)

        self.data_path = self.path/Path(f'model_{self.name_model}/steps_{self.steps}/overshoot_{self.overshoot}')

        self.atk = torchattacks.DeepFool(
                model=self.model,
                steps=self.steps,
                overshoot=self.overshoot
                )
        return

