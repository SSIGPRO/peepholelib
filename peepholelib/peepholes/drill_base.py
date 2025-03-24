# python stuff
import abc  

class DrillBase: 
    def __init__(self, **kwargs):
        self.path = kwargs['path']
        self.name = kwargs['name']

        # TODO: check if nl_class and n_features are not the same?
        self.nl_class = kwargs['nl_classifier'] if 'nl_classifier' in kwargs else None
        self.nl_model = kwargs['nl_model']
        self.n_features = kwargs['n_features']

        self.parser = kwargs['parser'] if 'parser' in kwargs else null_parser 
        self.parser_kwargs = kwargs['parser_kwargs'] if 'parser_kwargs' in kwargs and 'parser' in kwargs else dict() 

        self.bs = kwargs['batch_size'] if 'batch_size' in kwargs else '64'
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        # computed in fit()
        self._classifier = None

        # set in fit()
        self._cvs = None 

        # computer in compute_empirical_posteriors()
        self._empp = None

        # defined in save() or load()
        self._suffix = f'{self.name}.n_features={self.n_features}.nl_class={self.nl_class}.nl_model={self.nl_model}'
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

