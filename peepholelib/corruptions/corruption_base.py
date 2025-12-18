import abc

class CorruptionBase(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.severity = kwargs.get('severity', 1)
        if self.severity not in range(1, 6):
            raise ValueError(f'severity must be in [1, 5], got {self.severity}')

    @abc.abstractmethod
    def __call__(self, images, labels):
        raise NotImplementedError
