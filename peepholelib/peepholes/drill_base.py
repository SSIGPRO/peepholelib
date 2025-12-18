# python stuff
import abc 

class DrillBase(metaclass=abc.ABCMeta): 
    def __init__(self, **kwargs):
        self.path = kwargs['path']
        self.name = kwargs['name']
        self.target_module = kwargs['target_module']

        # number of classes in the NN model
        self.nl_model = kwargs['nl_model']

        # a.k.a corevector size
        self.n_features = kwargs['n_features']

        self.device = kwargs.get('device', 'cpu')

        # computed in fit()
        self._classifier = None
        self._cvs = None 

        return
    
    @abc.abstractmethod
    def __call__(self, *args, **kwds):
        pass
        
    @abc.abstractmethod
    def load(self, **kwargs):
        pass 

    @abc.abstractmethod
    def save(self, **kwargs):
        pass

    @abc.abstractmethod
    def fit(self, **kwargs):
        pass        

