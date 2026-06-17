import foolbox as fb

from .attack_base import AttackBase

_NORM_TO_ATTACK = {
    'L0':   fb.attacks.L0FMNAttack,
    'L1':   fb.attacks.L1FMNAttack,
    'L2':   fb.attacks.L2FMNAttack,
    'Linf': fb.attacks.LInfFMNAttack,
}

class ClampInputAttack:
    def __init__(self, attack, min_value=0.0, max_value=1.0):
        self.attack = attack
        self.model = attack.model
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, images, labels):
        images = images.clamp(self.min_value, self.max_value)
        adv_images = self.attack(images, labels)
        return adv_images.clamp(self.min_value, self.max_value)


def make_fmn_attack(**kwargs):
    attack = myFMN(**kwargs)
    attack.fmodel = fb.PyTorchModel(
        attack.model,
        bounds=(0, 1),
        device=attack.model.device,
    )
    return ClampInputAttack(attack)


class myFMN(AttackBase):
    """
    Fast Minimum-Norm (FMN) attack (Foolbox implementation).
    'Fast Minimum-Norm Adversarial Attacks through Adaptive Norm Constraints'
    [https://arxiv.org/abs/2102.12827]

    Arguments:
        model: model to attack.
        steps (int): number of optimization steps. (Default: 100)
        max_stepsize (float): initial (maximum) step size, cosine-annealed. (Default: 1.0)
        gamma (float): initial epsilon update step size, cosine-annealed to 0.001. (Default: 0.05)
        norm (str): perturbation norm — 'L0', 'L1', 'L2', or 'Linf'. (Default: 'Linf')

    Shape:
        - images: (N, C, H, W) with range [0, 1].
        - labels: (N,) with class indices.
        - output: (N, C, H, W).
    """

    def __init__(self, **kwargs):
        AttackBase.__init__(self, **kwargs)

        self.steps = kwargs.get('steps', 100)
        self.max_stepsize = kwargs.get('max_stepsize', 1.0)
        self.gamma = kwargs.get('gamma', 0.05)
        self.norm = kwargs.get('norm', 'Linf')

        if self.norm not in _NORM_TO_ATTACK:
            raise ValueError(
                f"norm must be one of {list(_NORM_TO_ATTACK)}, got '{self.norm}'"
            )

        self.fmodel = fb.PyTorchModel(
            self.model,
            bounds=(0, 1),
            device=self.model.device,
        )

        self.atk = _NORM_TO_ATTACK[self.norm](
            steps=self.steps,
            max_stepsize=self.max_stepsize,
            gamma=self.gamma,
        )

    def __call__(self, images, labels):
        images = images.to(self.model.device)
        labels = labels.to(self.model.device)
        _, clipped_advs, _ = self.atk(self.fmodel, images, labels, epsilons=None)
        return clipped_advs
