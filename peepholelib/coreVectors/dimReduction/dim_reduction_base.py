# General python stuff
import abc  

class DimReductionBase(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        return

    @abc.abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError()

