# python stuff
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import OrderedDict

# torch stuff
import torch
import torch.nn as nn

# our stuff
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.scores.model_output import ModelOutputScore
from peepholelib.scores.relu import RelUScore
from peepholelib.scores.doctor import DOCTORScore
from peepholelib.scores.feature_squeezing import FeatureSqueezingScore
from peepholelib.scores.vim import VIMScore
from peepholelib.scores.mahalanobis_plus import MahalanobisPlusScore
from peepholelib.scores.eps import EPSScore
from peepholelib.scores.protoclass import ProtoClassScore
from peepholelib.scores.cam import CAMLinScore, CAMExpScore
from peepholelib.scores.dmd import DMDBase, DMDPlus, DMDScore, DMDScoreConf

SEED = 0

#--------------------------------
# Fixtures
#--------------------------------

class FakeLoader():
    '''
    Stands in for `peepholelib.utils.ptd_wraps._StackedDS`: a key gives the whole column, a slice or a list of indices gives all the keys of those samples.
    '''
    def __init__(self, **kwargs):
        self._data = kwargs['data']
        self.len = len(next(iter(self._data.values())))
        return

    def __len__(self):
        return self.len

    def keys(self):
        return self._data.keys()

    def __getitem__(self, idx):
        if type(idx) == str:
            return self._data[idx]
        return {k: v[idx] for k, v in self._data.items()}

    def __getitems__(self, idx):
        return {k: v[idx] for k, v in self._data.items()}

class FakeDatasets():
    def __init__(self, **kwargs):
        self._dss = kwargs['dss']
        return

class FakePeepholes():
    def __init__(self, **kwargs):
        self._phs = kwargs['phs']
        return

    def get_conceptograms(self, **kwargs):
        loaders = kwargs['loaders']
        target_modules = kwargs.get('target_modules', None)

        ret = {}
        for ds_key in loaders:
            mks = target_modules if target_modules != None else list(self._phs[ds_key].keys())
            ret[ds_key] = torch.stack([self._phs[ds_key][mk] for mk in mks], dim=1)
        return ret

class FakeCoreVectors():
    def __init__(self, **kwargs):
        self._corevds = kwargs['corevds']
        return

class FakeDriller():
    '''
    Stands in for a fitted `DeepMahalanobisDistance` driller, as used by `DMDPlus`.
    '''
    def __init__(self, **kwargs):
        self.nl_model = kwargs['n_classes']
        self._means = kwargs['means']
        self._precision = kwargs['precision']
        return

def get_model(**kwargs):
    '''
    A real `ModelWrap` over a tiny model, so that hooks, activations and input gradients behave as in production.
    '''
    n_features = kwargs['n_features']
    n_hidden = kwargs['n_hidden']
    n_classes = kwargs['n_classes']

    torch.manual_seed(SEED)
    model = nn.Sequential(OrderedDict([
        ('flat', nn.Flatten()),
        ('fc', nn.Linear(n_features, n_hidden)),
        ('act', nn.ReLU()),
        ('head', nn.Linear(n_hidden, n_classes)),
        ]))
    return ModelWrap(model=model, device='cpu')

def get_datasets(**kwargs):
    '''
    Build the loaders with the inference values the scores read: `image`, `label`, `output`, `pred` and `result`.

    The fit loaders are built so that every class has at least one correctly classified sample (needed by `ProtoClassScore`) and so that both correctly and incorrectly classified samples are present (needed by `RelUScore` and `DMDScoreConf`).
    '''
    model = kwargs['model']
    loaders = kwargs['loaders']
    n_samples = kwargs['n_samples']
    n_classes = kwargs['n_classes']
    shape = kwargs['shape']

    torch.manual_seed(SEED)

    dss = {}
    for ds_key in loaders:
        images = torch.rand((n_samples,)+shape)
        with torch.no_grad():
            output = model(images)

        label = torch.arange(n_samples) % n_classes
        result = torch.zeros(n_samples, dtype=torch.long)
        # the first n_classes samples are correct, one per class, the rest alternate
        result[:n_classes] = 1
        result[n_classes::2] = 1
        pred = torch.where(result == 1, label, (label + 1) % n_classes)

        dss[ds_key] = FakeLoader(data={
            'image': images,
            'label': label,
            'output': output,
            'pred': pred,
            'result': result,
            })

    return FakeDatasets(dss=dss)

def get_peepholes(**kwargs):
    loaders = kwargs['loaders']
    n_samples = kwargs['n_samples']
    n_classes = kwargs['n_classes']
    target_modules = kwargs['target_modules']

    torch.manual_seed(SEED)
    phs = {
            ds_key: {mk: torch.rand(n_samples, n_classes) for mk in target_modules}
            for ds_key in loaders
            }
    return FakePeepholes(phs=phs)

def get_corevectors(**kwargs):
    loaders = kwargs['loaders']
    n_samples = kwargs['n_samples']
    cv_dim = kwargs['cv_dim']

    torch.manual_seed(SEED)
    corevds = {ds_key: {'cv': torch.rand(n_samples, cv_dim)} for ds_key in loaders}
    return FakeCoreVectors(corevds=corevds)

#--------------------------------
# Checks
#--------------------------------

def check_df(**kwargs):
    '''
    The returned object must be the scores `DataFrame`, with the expected columns, names, loaders and number of rows, and no NaN.
    '''
    df = kwargs['df']
    names = kwargs['names']
    n_rows = kwargs['n_rows']

    if list(df.columns) != ['dataset', 'score name', 'score value']:
        return f'columns are {list(df.columns)}'
    if sorted(set(df['score name'])) != sorted(names):
        return f'score names are {sorted(set(df["score name"]))}, expected {sorted(names)}'
    if len(df) != n_rows:
        return f'{len(df)} rows, expected {n_rows}'
    if df['score value'].isna().any():
        return 'contains NaN'
    return None

def check_values(**kwargs):
    '''
    Compare the recorded scores against a reference computed independently of the score class. Only used for the scores which have a closed form.
    '''
    df = kwargs['df']
    reference = kwargs['reference']
    loaders = kwargs['loaders']

    for ds_key in loaders:
        _ref = reference(ds_key)
        _got = torch.tensor(df[df['dataset'] == ds_key]['score value'].to_numpy(), dtype=torch.float32)

        if _got.shape != _ref.shape:
            return f'{ds_key}: {tuple(_got.shape)} scores, expected {tuple(_ref.shape)}'
        if not torch.allclose(_got, _ref.float(), atol=1e-5):
            return f'{ds_key}: values differ from the reference, max |diff| = {(_got - _ref.float()).abs().max():.2e}'
    return None

def check_covariance(**kwargs):
    '''
    The single-pass within-class scatter of the Mahalanobis fits must match the explicit centred covariance.
    '''
    score = kwargs['score']
    feats = kwargs['features']
    labels = kwargs['labels']
    n_classes = kwargs['n_classes']

    means = torch.zeros(n_classes, feats.shape[1])
    for c in range(n_classes):
        if (labels == c).any():
            means[c] = feats[labels == c].mean(dim=0)

    centered = feats.double() - means[labels].double()
    cov = (centered.t()@centered)/len(feats)
    precision = torch.linalg.pinv(cov, hermitian=True).float()

    if not torch.allclose(score._means, means, atol=1e-5):
        return f'means differ, max |diff| = {(score._means - means).abs().max():.2e}'
    if not torch.allclose(score._precision, precision, atol=1e-3):
        return f'precision differs, max |diff| = {(score._precision - precision).abs().max():.2e}'
    return None

def check_cache(**kwargs):
    '''
    Calling the score again must not add rows nor change the values.
    '''
    score = kwargs['score']
    score_kwargs = kwargs['score_kwargs']
    df = kwargs['df']

    _before = df['score value'].to_numpy().copy()
    _after = score.compute(**score_kwargs)['score value'].to_numpy()

    if len(_after) != len(_before):
        return f'{len(_after)} rows after re-calling, expected {len(_before)}'
    if (abs(_after - _before) > 1e-9).any():
        return 'values changed when re-calling'
    return None

def check_fitting_io(**kwargs):
    '''
    `fit()` must write `self._fit_file`, and a new instance must reproduce the very same scores from `load()` alone, with no `fit()`.
    '''
    score_class = kwargs['score_class']
    score_kwargs = kwargs['score_kwargs']
    fit_kwargs = kwargs.get('fit_kwargs', None)
    path = kwargs['path']

    score = score_class(path=path, **kwargs.get('init_kwargs', {}))

    if fit_kwargs == None:
        if score.load() != 1:
            return 'load() of a score which does not fit did not return 1'
        if score._save_fitting() != None:
            return '_save_fitting() of a score which does not fit returned a value'
        return None

    if score.load() != 0:
        return 'load() returned 1 before fitting'

    try:
        score.compute(**score_kwargs)
        return 'compute() did not raise before fit()'
    except RuntimeError:
        pass

    score.fit(**fit_kwargs)
    if not score._fit_file.exists():
        return f'fit() did not write {score._fit_file.name}'

    torch.manual_seed(SEED)
    _ref = score.compute(**score_kwargs)['score value'].to_numpy().copy()

    # drop the scores, keep the fitting: a new instance must score from load() alone
    score._file.unlink()
    _score = score_class(path=path, **kwargs.get('init_kwargs', {}))
    if _score.load() != 1:
        return 'load() did not return 1 with a saved fitting'

    torch.manual_seed(SEED)
    _got = _score.compute(**score_kwargs)['score value'].to_numpy()

    if len(_got) != len(_ref):
        return f'{len(_got)} rows after load(), expected {len(_ref)}'
    if (abs(_got - _ref) > 1e-5).any():
        return f'scores differ after load(), max |diff| = {abs(_got - _ref).max():.2e}'
    return None

def check_other_loader(**kwargs):
    '''
    A score fitted on one loader must score a different one without refitting, and the scores of a loader must not depend on which other loaders were computed with it.

    Three instances are compared: one scoring only the other loader, one scoring the fit loader and then the other one, and one scoring the other loader first. All three must give the same values for the other loader, and the fitted values must be untouched by `_compute()`.
    '''
    score_class = kwargs['score_class']
    score_kwargs = kwargs['score_kwargs']
    fit_kwargs = kwargs['fit_kwargs']
    init_kwargs = kwargs.get('init_kwargs', {})
    fit_key = kwargs['fit_key']
    other = kwargs['other_loader']
    path = kwargs['path']

    def _scores(loaders, tag):
        score = score_class(path=path/tag, **init_kwargs)
        score.fit(**fit_kwargs)
        _fitting = score._fit_file.read_bytes()

        torch.manual_seed(SEED)
        df = score.compute(**{**score_kwargs, 'loaders': loaders})

        _values = df[df['dataset'] == other]['score value'].to_numpy().copy()
        return _values, _fitting, score._fit_file.read_bytes()

    # only the other loader
    alone, fitting_before, fitting_after = _scores([other], 'alone')
    if fitting_before != fitting_after:
        return 'the fitted values changed while computing'

    # the fit loader first, then the other one
    after_fit_key, _, _ = _scores([fit_key, other], 'after_fit_key')

    # the other one first, then the fit loader
    before_fit_key, _, _ = _scores([other, fit_key], 'before_fit_key')

    if len(after_fit_key) != len(alone) or len(before_fit_key) != len(alone):
        return f'"{other}" got {len(alone)}/{len(after_fit_key)}/{len(before_fit_key)} rows depending on the loaders asked'
    if (abs(after_fit_key - alone) > 1e-5).any():
        return f'"{other}" changed when computed after "{fit_key}", max |diff| = {abs(after_fit_key - alone).max():.2e}'
    if (abs(before_fit_key - alone) > 1e-5).any():
        return f'"{other}" changed when computed before "{fit_key}", max |diff| = {abs(before_fit_key - alone).max():.2e}'
    return None

def check_resume(**kwargs):
    '''
    A run with every pair already recorded must be a no-op, not an error. The pairwise scores fit one model per pair and skip the pairs already in `self.df`, so on such a run `fit()` has nothing to fit and leaves the fitted values empty: that must not be mistaken for a score which was never fitted.

    Checked with the fitting file in place, where `load()` restores it, and with the fitting file deleted, where `fit()` is called and finds nothing pending.
    '''
    score_class = kwargs['score_class']
    score_kwargs = kwargs['score_kwargs']
    fit_kwargs = kwargs['fit_kwargs']
    init_kwargs = kwargs.get('init_kwargs', {})
    path = kwargs['path']

    # first run: fit and score every pair
    score = score_class(path=path/'resume', **init_kwargs)
    score.fit(**fit_kwargs)
    _ref = score.compute(**score_kwargs)['score value'].to_numpy().copy()

    for _drop_fitting in [False, True]:
        _score = score_class(path=path/'resume', **init_kwargs)
        if _drop_fitting:
            _score._fit_file.unlink()

        _tag = 'without the fitting file' if _drop_fitting else 'with the fitting file'
        if not _score.load():
            _score.fit(**fit_kwargs)

        try:
            _got = _score.compute(**score_kwargs)['score value'].to_numpy()
        except RuntimeError as e:
            return f'resuming {_tag} raised: {e}'

        if len(_got) != len(_ref):
            return f'resuming {_tag} gave {len(_got)} rows, expected {len(_ref)}'
        if (abs(_got - _ref) > 1e-9).any():
            return f'resuming {_tag} changed the scores'
    return None

def check_extra_pair(**kwargs):
    '''
    The pairwise scores are fitted per pair. Fitting a second negative loader later must add it without touching the pair fitted first, and without refitting it.
    '''
    score_class = kwargs['score_class']
    score_kwargs = kwargs['score_kwargs']
    fit_kwargs = kwargs['fit_kwargs']
    init_kwargs = kwargs.get('init_kwargs', {})
    extra_neg_loaders = kwargs['extra_neg_loaders']
    path = kwargs['path']

    score = score_class(path=path/'pairs', **init_kwargs)

    torch.manual_seed(SEED)
    score.fit(**fit_kwargs)
    df = score.compute(**score_kwargs)

    _names = sorted(set(df['score name']))
    _first = df[df['score name'] == _names[0]]['score value'].to_numpy().copy()

    # a second pair, fitted and scored on top of the first
    torch.manual_seed(SEED)
    score.fit(**{**fit_kwargs, 'neg_loaders': extra_neg_loaders})
    df = score.compute(**score_kwargs)

    _new_names = sorted(set(df['score name']))
    if len(_new_names) != len(_names) + len(extra_neg_loaders):
        return f'score names are {_new_names}, expected {len(_names) + len(extra_neg_loaders)} of them'

    _again = df[df['score name'] == _names[0]]['score value'].to_numpy()
    if len(_again) != len(_first):
        return f'"{_names[0]}" has {len(_again)} rows after the second pair, expected {len(_first)}'
    if (abs(_again - _first) > 1e-9).any():
        return f'"{_names[0]}" changed when the second pair was added'
    return None

def run(**kwargs):
    '''
    Run one score end to end: compute, cache and fitting round trip.
    '''
    name = kwargs['name']
    score_class = kwargs['score_class']
    score_kwargs = kwargs['score_kwargs']
    init_kwargs = kwargs.get('init_kwargs', {})
    fit_kwargs = kwargs.get('fit_kwargs', None)
    names = kwargs['names']
    n_rows = kwargs['n_rows']
    path = kwargs['path']
    reference = kwargs.get('reference', None)
    fit_check = kwargs.get('fit_check', None)
    other_loader = kwargs.get('other_loader', None)
    extra_neg_loaders = kwargs.get('extra_neg_loaders', None)

    with TemporaryDirectory(dir=path) as tmp:
        score = score_class(path=Path(tmp)/'scores', **init_kwargs)
        if fit_kwargs != None:
            score.fit(**fit_kwargs)

        torch.manual_seed(SEED)
        df = score.compute(**score_kwargs)

        errors = [
                check_df(df=df, names=names, n_rows=n_rows),
                check_cache(score=score, score_kwargs=score_kwargs, df=df),
                ]
        if reference != None:
            errors.append(check_values(df=df, reference=reference, loaders=score_kwargs['loaders']))
        if fit_check != None:
            errors.append(fit_check(score))

    with TemporaryDirectory(dir=path) as tmp:
        errors.append(check_fitting_io(
            score_class = score_class,
            init_kwargs = init_kwargs,
            score_kwargs = score_kwargs,
            fit_kwargs = fit_kwargs,
            path = Path(tmp)/'scores',
            ))

    if other_loader != None:
        with TemporaryDirectory(dir=path) as tmp:
            errors.append(check_other_loader(
                score_class = score_class,
                init_kwargs = init_kwargs,
                score_kwargs = score_kwargs,
                fit_kwargs = fit_kwargs,
                fit_key = fit_kwargs['fit_key'],
                other_loader = other_loader,
                path = Path(tmp),
                ))

    if extra_neg_loaders != None:
        with TemporaryDirectory(dir=path) as tmp:
            errors.append(check_resume(
                score_class = score_class,
                init_kwargs = init_kwargs,
                score_kwargs = score_kwargs,
                fit_kwargs = fit_kwargs,
                path = Path(tmp),
                ))

        with TemporaryDirectory(dir=path) as tmp:
            errors.append(check_extra_pair(
                score_class = score_class,
                init_kwargs = init_kwargs,
                score_kwargs = score_kwargs,
                fit_kwargs = fit_kwargs,
                extra_neg_loaders = extra_neg_loaders,
                path = Path(tmp),
                ))

    errors = [e for e in errors if e != None]
    status = 'ok' if len(errors) == 0 else 'FAIL'
    _extra = ' +other-loader' if other_loader != None else (' +resume +extra-pair' if extra_neg_loaders != None else '')
    print(f'{name:24s} {status:5s} rows={n_rows:<5d} fit={"yes" if fit_kwargs != None else "no ":3s}{_extra}' +
          ('' if len(errors) == 0 else '\n' + '\n'.join(f'{"":26s}- {e}' for e in errors)))
    return len(errors)

if __name__ == '__main__':
    n_samples = 48
    n_classes = 4
    n_hidden = 8
    shape = (3, 4, 4)
    n_features = shape[0]*shape[1]*shape[2]
    target_modules = ['m0', 'm1']
    output_layer = 'head'

    loaders = ['train', 'val', 'test', 'ood-val', 'ood-test']
    score_loaders = ['test', 'ood-test']
    neg_loaders = {'ood-test': ['ood-val']}

    model = get_model(n_features=n_features, n_hidden=n_hidden, n_classes=n_classes)
    dss = get_datasets(model=model, loaders=loaders, n_samples=n_samples, n_classes=n_classes, shape=shape)
    phs = get_peepholes(loaders=loaders, n_samples=n_samples, n_classes=n_classes, target_modules=target_modules)
    cvs = get_corevectors(loaders=loaders, n_samples=n_samples, cv_dim=n_hidden)

    torch.manual_seed(SEED)
    driller = FakeDriller(
            n_classes = n_classes,
            means = torch.rand(n_classes, n_hidden),
            precision = torch.eye(n_hidden),
            )

    # a valid diffusion score function for a standard normal, used by EPSScore
    score_fn = lambda x_t, t: -x_t

    detector = lambda images: images.flatten(1).mean(dim=1)

    #--------------------------------
    # references, computed independently of the score classes
    #--------------------------------
    def ref_model_output(_type):
        def _ref(ds_key):
            output = dss._dss[ds_key]['output']
            if _type == 'MSP': return output.softmax(dim=-1).max(dim=-1).values
            if _type == 'MaxLogit': return output.max(dim=-1).values
            if _type == 'Energy': return torch.logsumexp(output, dim=-1)
            probs = output.softmax(dim=-1)
            entropy = -(probs*torch.log(probs.clamp(min=1e-8))).sum(dim=-1)
            return 1.0 - entropy/torch.log(torch.tensor(float(n_classes)))
        return _ref

    def ref_doctor(ds_key):
        probs = dss._dss[ds_key]['output'].softmax(dim=-1)
        return (probs**2).sum(dim=-1)

    def ref_feature_squeezing(ds_key):
        return (2 - detector(dss._dss[ds_key]['image']))/2

    def ref_cam_lin(ds_key):
        h = sum(phs._phs[ds_key][mk] for mk in target_modules)/len(target_modules)
        pred = dss._dss[ds_key]['pred']
        return 1 - h.gather(1, pred.unsqueeze(1)).squeeze(1)

    def ref_dmd_plus(ds_key):
        x = cvs._corevds[ds_key]['cv']
        x = x/torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        gaussian_score = torch.zeros(x.shape[0], n_classes)
        for c in range(n_classes):
            zero_f = x - driller._means[c].view(1, -1)
            gaussian_score[:, c] = -(zero_f@driller._precision@zero_f.t()).diag()
        return gaussian_score.max(dim=1)[0]

    def get_activations(ds_key, normalize):
        model.set_target_modules(target_modules=[output_layer])
        model.set_activations(save_input=True, save_output=False)
        with torch.no_grad():
            model(dss._dss[ds_key]['image'])
        acts = model._acts['in_activations'][output_layer]
        acts = acts.view(acts.shape[0], -1).clone()
        model.set_activations(save_input=False, save_output=False)
        return torch.nn.functional.normalize(acts, p=2, dim=1) if normalize else acts

    def fit_check_maha(normalize):
        return lambda score: check_covariance(
                score = score,
                features = get_activations('train', normalize),
                labels = dss._dss['train']['label'],
                n_classes = n_classes,
                )

    n_pair = 2*n_samples          # pairwise scores record the positive and the negative loader
    n_both = len(score_loaders)*n_samples

    with TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir)
        n_failed = 0

        #--------------------------------
        # scores of the model outputs
        #--------------------------------
        for _type in ['MSP', 'MaxLogit', 'Energy', 'PE']:
            n_failed += run(
                    name = f'ModelOutputScore[{_type}]',
                    score_class = ModelOutputScore,
                    init_kwargs = {'type': _type},
                    score_kwargs = {'datasets': dss, 'loaders': score_loaders},
                    reference = ref_model_output(_type),
                    names = [_type],
                    n_rows = n_both,
                    path = path,
                    )

        n_failed += run(
                name = 'RelUScore',
                score_class = RelUScore,
                fit_kwargs = {'datasets': dss, 'fit_key': 'val'},
                score_kwargs = {'datasets': dss, 'loaders': score_loaders},
                other_loader = 'ood-test',
                names = ['Rel-U'],
                n_rows = n_both,
                path = path,
                )

        for _mag in [0.0, 0.01]:
            n_failed += run(
                    name = f'DOCTORScore[mag={_mag}]',
                    score_class = DOCTORScore,
                    score_kwargs = {'datasets': dss, 'model': model, 'loaders': score_loaders,
                                    'magnitude': _mag, 'batch_size': 16, 'n_threads': 0},
                    reference = ref_doctor if _mag == 0.0 else None,
                    names = ['DOCTOR'],
                    n_rows = n_both,
                    path = path,
                    )

        #--------------------------------
        # scores of the inputs
        #--------------------------------
        n_failed += run(
                name = 'FeatureSqueezingScore',
                score_class = FeatureSqueezingScore,
                score_kwargs = {'datasets': dss, 'loaders': score_loaders, 'detector': detector,
                                'batch_size': 16, 'n_threads': 0},
                reference = ref_feature_squeezing,
                names = ['Feature-Squeezing'],
                n_rows = n_both,
                path = path,
                )

        n_failed += run(
                name = 'EPSScore',
                score_class = EPSScore,
                fit_kwargs = {'datasets': dss, 'score_fn': score_fn, 'fit_key': 'train',
                              'T_star': 2, 'batch_size': 16, 'n_threads': 0},
                score_kwargs = {'datasets': dss, 'score_fn': score_fn, 'loaders': score_loaders,
                                'batch_size': 16, 'n_threads': 0},
                # no `other_loader`: `compute_eps()` samples noise, so the scores of a loader
                # depend on how much of the RNG the loaders computed before it consumed
                names = ['EPS'],
                n_rows = n_both,
                path = path,
                )

        #--------------------------------
        # scores of the activations
        #--------------------------------
        n_failed += run(
                name = 'VIMScore',
                score_class = VIMScore,
                fit_kwargs = {'model': model, 'datasets': dss, 'output_layer': output_layer,
                              'fit_key': 'train', 'batch_size': 16, 'n_threads': 0},
                score_kwargs = {'model': model, 'datasets': dss, 'output_layer': output_layer,
                                'loaders': score_loaders, 'batch_size': 16, 'n_threads': 0},
                other_loader = 'ood-test',
                names = ['ViM'],
                n_rows = n_both,
                path = path,
                )

        for _name, _class in [('DMDBase', DMDBase), ('MahalanobisPlusScore', MahalanobisPlusScore)]:
            n_failed += run(
                    name = _name,
                    score_class = _class,
                    fit_kwargs = {'model': model, 'datasets': dss, 'output_layer': output_layer,
                                  'n_classes': n_classes, 'fit_key': 'train', 'batch_size': 16, 'n_threads': 0},
                    score_kwargs = {'model': model, 'datasets': dss, 'output_layer': output_layer,
                                    'loaders': score_loaders, 'batch_size': 16, 'n_threads': 0},
                    other_loader = 'ood-test',
                    fit_check = fit_check_maha(_class is MahalanobisPlusScore),
                    names = [_class(path=path/'_probe').name],
                    n_rows = n_both,
                    path = path,
                    )

        #--------------------------------
        # scores of the corevectors and peepholes
        #--------------------------------
        n_failed += run(
                name = 'DMDPlus',
                score_class = DMDPlus,
                score_kwargs = {'coreavg': cvs, 'driller': driller, 'layer': 'cv',
                                'loaders': score_loaders, 'device': 'cpu'},
                reference = ref_dmd_plus,
                names = ['dmd_plus'],
                n_rows = n_both,
                path = path,
                )

        n_failed += run(
                name = 'ProtoClassScore',
                score_class = ProtoClassScore,
                fit_kwargs = {'datasets': dss, 'peepholes': phs, 'fit_key': 'train',
                              'target_modules': target_modules, 'proto_threshold': 0.0},
                score_kwargs = {'datasets': dss, 'peepholes': phs, 'loaders': score_loaders,
                                'target_modules': target_modules},
                other_loader = 'ood-test',
                names = ['Proto-Class'],
                n_rows = n_both,
                path = path,
                )

        n_failed += run(
                name = 'CAMLinScore',
                score_class = CAMLinScore,
                score_kwargs = {'datasets': dss, 'peepholes': phs, 'loaders': score_loaders,
                                'target_modules': target_modules},
                reference = ref_cam_lin,
                names = ['CAM-lin'],
                n_rows = n_both,
                path = path,
                )

        n_failed += run(
                name = 'CAMExpScore',
                score_class = CAMExpScore,
                fit_kwargs = {'datasets': dss, 'peepholes': phs, 'pos_loader_train': 'val',
                              'neg_loaders': neg_loaders, 'target_modules': target_modules},
                score_kwargs = {'datasets': dss, 'peepholes': phs, 'pos_loader_test': 'test',
                                'target_modules': target_modules},
                extra_neg_loaders = {'ood-val': ['ood-test']},
                names = ['CAM-ood-test'],
                n_rows = n_pair,
                path = path,
                )

        n_failed += run(
                name = 'DMDScore',
                score_class = DMDScore,
                fit_kwargs = {'peepholes': phs, 'pos_loader_train': 'val',
                              'neg_loaders': neg_loaders, 'target_modules': target_modules},
                score_kwargs = {'peepholes': phs, 'pos_loader_test': 'test',
                                'target_modules': target_modules},
                extra_neg_loaders = {'ood-val': ['ood-test']},
                names = ['DMD-ood-test'],
                n_rows = n_pair,
                path = path,
                )

        n_failed += run(
                name = 'DMDScoreConf',
                score_class = DMDScoreConf,
                fit_kwargs = {'peepholes': phs, 'datasets': dss, 'fit_key': 'val',
                              'target_modules': target_modules},
                score_kwargs = {'peepholes': phs, 'loaders': score_loaders,
                                'target_modules': target_modules},
                other_loader = 'ood-test',
                names = ['DMD-in'],
                n_rows = n_both,
                path = path,
                )

    print()
    print('all scores passed' if n_failed == 0 else f'{n_failed} check(s) failed')
