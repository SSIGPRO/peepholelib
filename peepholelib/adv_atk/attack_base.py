# general python stuff
import abc 

def from_tensorDict(data, key_list):
    return {k: data[k] for k in key_list} 

class AttackBase(metaclass=abc.ABCMeta):
    
    def __init__(self, **kwargs):
        self.model = kwargs.get('model')
        return
    
    @abc.abstractmethod
    def __call__(self, images, labels):
        raise NotImplementedError
