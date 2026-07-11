# python stuff
from pathlib import Path

# torch stuff
import torch
from peepholelib.peepholes.drill_base import DrillBase

class MRC(DrillBase):
    '''
    Multi-Range Coverage (MRC) driller, as described in Section 5.2 of "Increasing the Confidence of Deep Neural Networks by Coverage Analysis" (Rossolini et al., IEEE TSE 2023).

    The driller is associated with a single layer (`target_module`, set in DrillBase). For each model class, the trusted range of every neuron's output is split into `Q` equally-sized sub-ranges, and the frequency with which the trusted samples fall into each sub-range is recorded (Algorithm 3, "Aggregation Algorithm of MRC"). At inference time (Algorithm 4, "Confidence Evaluation Algorithm of MRC"), `__call__()` returns, for each model class, the cost `eta` accumulated over the neurons of this layer, instead of the confidence value `c = exp(-eta*ln(2)/tau)` reported in the paper, so that the thresholding/confidence mapping can be applied downstream. During `fit()`, only correctly classified samples (`'result' == True`) whose maximum model output exceeds `confidence_threshold` are used to build the signature.
    '''
    def __init__(self, **kwargs):
        DrillBase.__init__(self, **kwargs)

        # number of sub-ranges (Q in the paper)
        self.Q = kwargs.get('Q', 1)

        # minimum model confidence (max output) for a sample to be included in fit()
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.9)

        # normalize the peepholes by the total number of neurons, so their range lies in [0,1]
        self.normalize = kwargs.get('normalize', True)

        self.reducer = kwargs['reducer']
        self.cv_parser = self.reducer.parser

        # computed in fit()
        # v_min, v_max, delta: (nl_model, n_features)
        # lam (lambda in the paper): (nl_model, Q, n_features)
        self._v_min = None
        self._v_max = None
        self._delta = None
        self._lam = None

        # used in save() and load()
        self._mrc_folder = self.path/self.name
        self._signature_path = self._mrc_folder/'signature.pt'
        return

    def fit(self, **kwargs):
        '''
        Compute the DNN Signature for MRC (Algorithm 3).

        Args:
        - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): Parsed datasets respective the `coreVectors`. Must contain `'result'` (correct classification flag) and `'output'` (raw model outputs) keys.
        - corevectors (peepholelib.coreVectors.coreVectors.CoreVectors): Corevectors respective the `datasets`.
        - loader (str): Which loader used for fitting MRC, usually 'train'. Defaults to 'train'.
        - label_key (str): Key used to read the class label from the dataset. Defaults to 'label'.
        '''
        _dss = kwargs['datasets']
        _cvs = kwargs['corevectors']
        loader = kwargs.get('loader', 'train')
        label_key = kwargs.get('label_key', 'label')

        dss = _dss._dss[loader]
        cvs = self.cv_parser(cvs=_cvs._corevds[loader][self.target_module])

        if cvs.shape[1] != self.n_features:
            raise RuntimeError(f'Something is weird...\n Data has shape {cvs.shape} after parsing corevectors with the parser {self.cv_parser}\nWhile n_features={self.n_features} was passed during construction.')

        cvs = cvs.to(self.device)
        labels = dss[:][label_key].int().to(self.device)

        # keep only correctly classified samples above the confidence threshold
        results = dss[:]['result'].to(self.device)
        confidence = dss[:]['output'].softmax(dim=1).max(dim=1).values.to(self.device)
        mask = results & (confidence >= self.confidence_threshold)
        cvs = cvs[mask]
        labels = labels[mask]

        self._v_min = torch.zeros(self.nl_model, self.n_features, device=self.device)
        self._v_max = torch.zeros(self.nl_model, self.n_features, device=self.device)
        self._delta = torch.zeros(self.nl_model, self.n_features, device=self.device)
        self._lam = torch.zeros(self.nl_model, self.Q, self.n_features, device=self.device)

        for i in range(self.nl_model):
            data_i = cvs[labels == i]
            n_i = data_i.shape[0]

            if n_i == 0:
                raise RuntimeError(f'No samples for class {i} in loader "{loader}". Cannot compute MRC signature for this class.')

            v_min = torch.amin(data_i, dim=0)
            v_max = torch.amax(data_i, dim=0)
            delta = (v_max - v_min)/self.Q

            # avoid division by zero for neurons that are constant over S_i
            delta_safe = delta.clone()
            delta_safe[delta_safe == 0] = 1

            q_idx = (torch.ceil((data_i - v_min)/delta_safe).long() - 1).clamp(min=0, max=self.Q - 1)

            lam = torch.zeros(self.Q, self.n_features, device=self.device)
            lam.scatter_add_(0, q_idx, torch.ones_like(data_i))
            lam /= n_i

            self._v_min[i] = v_min
            self._v_max[i] = v_max
            self._delta[i] = delta
            self._lam[i] = lam
        return

    def __call__(self, **kwargs):
        '''
        Confidence Evaluation Algorithm of MRC (Algorithm 4), returning the accumulated cost `eta` for each model class on this layer. For each neuron, the cost is `1 - lambda_q*` if the activation is within the trusted range `[v_min, v_max]`, and `1` otherwise.

        Args:
        - cvs (torch.Tensor): Batch of corevectors for this layer, will be parsed with self.cv_parser (see __init__()).

        Returns:
        - eta (torch.Tensor): Tensor of shape (n_samples, nl_model) with the accumulated cost for each class.
        '''
        if self._v_min is None:
            raise RuntimeError('MRC signature not computed. Please run fit() or load() first.')

        cvs = kwargs['cvs']
        data = self.cv_parser(cvs=cvs).to(self.device)

        n_samples = data.shape[0]
        eta = torch.zeros(n_samples, self.nl_model, device=self.device)

        for i in range(self.nl_model):
            v_min = self._v_min[i]
            v_max = self._v_max[i]
            delta = self._delta[i]
            lam = self._lam[i] # (Q, n_features)

            # (n_samples, n_features)
            out_of_range = (data < v_min) | (data > v_max)

            delta_safe = delta.clone()
            delta_safe[delta_safe == 0] = 1

            q_idx = torch.ceil((data - v_min)/delta_safe).long()
            # 0-indexed, clamped to valid range
            q_idx = (q_idx - 1).clamp(min=0, max=self.Q - 1)

            lam_vals = lam[q_idx, torch.arange(self.n_features, device=self.device)]

            cost = torch.where(out_of_range, torch.ones_like(data), 1-lam_vals)
            eta[:, i] = cost.sum(dim=1)

            if self.normalize:
                eta[:, i] /= self.n_features

        return eta

    def save(self, **kwargs):
        self._mrc_folder.mkdir(parents=True, exist_ok=True)

        torch.save({
            'v_min': self._v_min.detach().cpu(),
            'v_max': self._v_max.detach().cpu(),
            'delta': self._delta.detach().cpu(),
            'lam': self._lam.detach().cpu(),
            }, self._signature_path)
        return

    def load(self, **kwargs):
        if self._signature_path.exists():
            signature = torch.load(self._signature_path, weights_only=True)
            self._v_min = signature['v_min'].to(self.device)
            self._v_max = signature['v_max'].to(self.device)
            self._delta = signature['delta'].to(self.device)
            self._lam = signature['lam'].to(self.device)
            ok = True
        else:
            ok = False
        return ok
