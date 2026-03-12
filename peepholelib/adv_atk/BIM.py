# torch stuff
import torch
import torchattacks

# our stuff
from .attack_base import AttackBase


class BIMStoreTarget(torchattacks.BIM):
    def get_target_label(self, inputs, labels=None):
        """
        same as PGD to store targets
        """
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

        self.eps = kwargs.get("eps", 8/255)
        self.alpha = kwargs.get("alpha", 2/255)
        self.steps = kwargs.get("steps", 300)
        self.mode = kwargs.get("mode", "random")
        self.target_class = kwargs.get("target_class", 5)

        self.atk = BIMStoreTarget(
            model=self.model._model,
            eps=self.eps,
            alpha=self.alpha,
            steps=self.steps,
        )

        if self.mode == "random":
            self.atk.set_mode_targeted_random(quiet=False)

        elif self.mode == "least-likely":
            self.atk.set_mode_targeted_least_likely(kth_min=1, quiet=False)

        elif self.mode == "fixed":
            tc = int(self.target_class)

            def fixed_target_fn(inputs, labels):
                y_t = torch.full_like(labels, tc)
                same = (y_t == labels)
                if same.any():
                    with torch.no_grad():
                        num_classes = self.atk.get_output_with_eval_nograd(inputs).shape[1]
                    y_t[same] = (y_t[same] + 1) % num_classes
                return y_t

            self.atk.set_mode_targeted_by_function(fixed_target_fn, quiet=False)

    def __call__(self, images, labels):
        return self.atk(images, labels)