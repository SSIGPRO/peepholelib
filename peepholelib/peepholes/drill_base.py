# python stuff
import abc 

class DrillBase: 
    def __init__(self, **kwargs):
        self.path = kwargs['path']
        self.name = kwargs['name']

        # number of classes in the NN model
        self.nl_model = kwargs['nl_model']

        # a.k.a corevector size
        self.n_features = kwargs['n_features']

        # number of classes in classifier, e.g., number of clusters
        self.nl_class = kwargs['nl_classifier'] if 'nl_classifier' in kwargs else None


        self.parser = kwargs['parser']
        self.parser_kwargs = kwargs['parser_kwargs'] if 'parser_kwargs' in kwargs else {} 

        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        # computed in fit()
        self._classifier = None

        # set in fit()
        self._cvs = None 

        # computer in compute_empirical_posteriors()
        self._empp = None

        # used in save() or load()
        self._suffix = f'.nl_model={self.nl_model}.n_features={self.n_features}'
        if not (self.nl_class is None):
            self._suffix += f'.nl_class={self.nl_class}'

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

