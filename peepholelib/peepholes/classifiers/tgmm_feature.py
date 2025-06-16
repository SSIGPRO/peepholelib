# our stuff
from .classifier_base import ClassifierBase

# torch stuff
import torch
from torch.nn.functional import softmax as sm

# https://github.com/CSOgroup/torchgmm/tree/main
from torchgmm.bayes import GaussianMixture as tGMM

import logging
logging.getLogger('pytorch_lightning.utilities.rank_zero').setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.CRITICAL)
class GMM_feature(ClassifierBase):  # windowed GMM
    def __init__(self, **kwargs):
        self.window_size = kwargs.pop('window_size') if 'window_size' in kwargs else 3
        cls_kwargs = kwargs.pop('cls_kwargs') if 'cls_kwargs' in kwargs else {}
        ClassifierBase.__init__(self, **kwargs)

        self._classifiers = []  # list of tGMM instances, length = n_windows after fit

        self._cls_kwargs = cls_kwargs

        self._clas_path = self.path / (self.name + '.GMM' + self._suffix)
        self._empp_file = self._clas_path / f'empp_{self.label_key}.pt'
        self._meta_file = self._clas_path / 'meta.pt'

        return

    def _make_classifier(self):
        trainer_params = dict(
            num_nodes = 1,
            max_epochs = 50000,
            accelerator = self.device.type,
            devices = [self.device.index],
            enable_progress_bar = False
        )
        kwargs = dict(num_components=self.nl_class, **self._cls_kwargs, trainer_params=trainer_params)
        return tGMM(**kwargs)

    def fit(self, **kwargs):
        """
        Fit windowed GMM(s).

        Required kwargs:
         - corevectors (TensorDict-like): object used with parser(cvs=...)
        Optional:
         - loader: loader name (default 'train')
         - verbose: bool
        """
        _cvs = kwargs['corevectors']
        loader = kwargs.get('loader', 'train')
        verbose = kwargs.get('verbose', False)

        cvs = _cvs._corevds[loader]
        if verbose: print('\n ---- Windowed GMM classifier (window_size=%d)\n' % self.window_size)

        # parse the dataset into a data tensor (N, D)
        data = self.parser(cvs=cvs)
        # parser expected to return a tensor-like with shape [N, features]
        if data.shape[1] != self.n_features:
            raise RuntimeError(
                f'Something is weird...\n Data has shape {data.shape} after parsing corevectors with the parser {self.parser}\nWhile n_features={self.n_features} was passed during construction.'
            )

        N, D = data.shape
        ws = int(self.window_size)
        n_windows = (D + ws - 1) // ws

        self._classifiers = []
        for wi in range(n_windows):
            s = wi * ws
            e = min((wi + 1) * ws, D)
            win_data = data[:, s:e].clone().detach().to(self.device)

            if verbose: print(f'Fitting window {wi+1}/{n_windows} dims {s}:{e}')

            clf = self._make_classifier()

            # retry loop to avoid NaNs as in original
            converged = False
            while not converged:
                clf.fit(win_data)
                with torch.no_grad():
                    check = clf.predict_proba(win_data[0:1])
                    converged = not torch.isnan(check).any()
                if verbose and (not converged):
                    print(f'GMM window {wi} failed, retrying.')

            self._classifiers.append(clf)

        meta = {
            'window_size': ws,
            'n_windows': n_windows,
            'n_features': D
        }
        self._clas_path.mkdir(parents=True, exist_ok=True)
        torch.save(meta, self._meta_file)

        return

    def classifier_probabilities(self, **kwargs):
        data = kwargs['data']
        N, D = data.shape

        if not self._classifiers:
            raise RuntimeError('No classifiers available. Did you run fit() or load()?')

        probs_list = []
        ws = int(self.window_size)

        for wi, clf in enumerate(self._classifiers):
            s = wi * ws
            e = min((wi + 1) * ws, D)
            win_data = data[:, s:e].to(self.device)
            p = clf.predict_proba(win_data)  # [N, G]
            probs_list.append(p.to(self.device))

        # [W, N, G]
        return torch.stack(probs_list, dim=0)


    def predict(self, data):
        probs = self.classifier_probabilities(data=data)
        return probs.argmax(dim=1)

    def save(self, **kwargs):
        self._clas_path.mkdir(parents=True, exist_ok=True)
        for wi, clf in enumerate(self._classifiers):
            wdir = self._clas_path / f'window_{wi}'
            wdir.mkdir(parents=True, exist_ok=True)
            clf.save(wdir)
        if hasattr(self, '_meta_file') and self._meta_file.exists():
            pass 
        super().save()
        return

    def load(self, **kwargs):
        # load meta
        if not self._meta_file.exists():
            raise RuntimeError(f'Meta file not found at {self._meta_file}. Cannot load windowed classifiers.')
        meta = torch.load(self._meta_file)
        ws = meta['window_size']
        n_windows = meta['n_windows']
        self.window_size = ws

        self._classifiers = []
        for wi in range(n_windows):
            wdir = self._clas_path / f'window_{wi}'
            clf = tGMM.load(wdir)
            self._classifiers.append(clf)

        # load empirical posteriors if present
        super().load()
        return

    def load_without_empp(self, **kwargs):
        # similar to load but avoid calling super().load to skip _empp
        if not self._meta_file.exists():
            raise RuntimeError(f'Meta file not found at {self._meta_file}. Cannot load windowed classifiers.')
        meta = torch.load(self._meta_file)
        ws = meta['window_size']
        n_windows = meta['n_windows']
        self.window_size = ws

        self._classifiers = []
        for wi in range(n_windows):
            wdir = self._clas_path / f'window_{wi}'
            clf = tGMM.load(wdir)
            self._classifiers.append(clf)

        return
