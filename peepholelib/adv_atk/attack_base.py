# general python stuff
import abc 
import torch

def from_tensorDict(data, key_list):
    return {k: data[k] for k in key_list} 

class AttackBase(metaclass=abc.ABCMeta):
    
    def __init__(self, **kwargs):
        self.model = kwargs.get('model')
        self.mode = kwargs.get('mode', 'random')
        self.target_class = kwargs.get('target_class', 5)
        return

    def _fixed_target_fn(self, inputs, labels):
        """
        Build fixed target labels for targeted attacks.
        """
        tc = int(self.target_class)
        y_t = torch.full_like(labels, tc)

        same = (y_t == labels)
        if same.any():
            with torch.no_grad():
                num_classes = self.atk.get_output_with_eval_nograd(inputs).shape[1]
            y_t[same] = (y_t[same] + 1) % num_classes

        return y_t
        
    def _set_targeted_mode(self):
        """
        Configure attack target selection
        """
        if self.mode == "random":
            self.atk.set_mode_targeted_random(quiet=False)

        elif self.mode == "least-likely":
            self.atk.set_mode_targeted_least_likely(kth_min=1, quiet=False)

        elif self.mode == "fixed":
            self.atk.set_mode_targeted_by_function(self._fixed_target_fn, quiet=False)
    
    @abc.abstractmethod
    def __call__(self, images, labels):
        raise NotImplementedError
