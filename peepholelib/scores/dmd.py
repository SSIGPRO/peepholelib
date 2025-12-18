from math import floor, ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensordict import PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from peepholelib.scores.score import Score

def _find_classification_head(model_wrap):
    last_name = None
    for name, module in model_wrap._model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
    if last_name is None:
        raise RuntimeError('No nn.Linear module found in the model.')
    return last_name

def DMD_base_fit(**kwargs):
    '''
    Compute class-conditional means and shared precision matrix from training activations of the penultimate layer (Lee et al., https://arxiv.org/abs/1807.03888) and save them to disk.

    Args:
    - model (peepholelib.models.model_wrap.ModelWrap): wrapped model.
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - fit_key (str): loader key used for fitting. Defaults to 'train'.
    - n_classes (int): number of classes.
    - stats_path (str | Path): path where the statistics .pt file will be saved.
    - batch_size (int): batch size for the dataloader. Defaults to 128.
    - n_threads (int): number of dataloader workers. Defaults to 1.
    - verbose (bool): print progress. Defaults to False.
    '''
    model = kwargs['model']
    datasets = kwargs['datasets']
    fit_key = kwargs.get('fit_key', 'train')
    n_classes = kwargs['n_classes']
    stats_path = Path(kwargs['stats_path'])
    bs = kwargs.get('batch_size', 128)
    n_threads = kwargs.get('n_threads', 1)
    verbose = kwargs.get('verbose', False)

    if stats_path.exists():
        if verbose:
            print(f'DMD statistics already exist at {stats_path}. Skipping fit.')
        return

    device = model.device
    layer = _find_classification_head(model)

    model.set_target_modules(target_modules=[layer])
    model.set_activations(save_input=True, save_output=False)
    model._model.eval()

    dl = DataLoader(
        datasets._dss[fit_key],
        batch_size=bs,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=n_threads,
        pin_memory=(device != 'cpu'),
    )

    n_samples = len(datasets._dss[fit_key])
    with torch.no_grad():
        sample = datasets._dss[fit_key][0:1]
        _ = model(sample['image'].to(device))
        act0 = model._acts['in_activations'][layer]
        n_features = act0.view(act0.shape[0], -1).shape[1]

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'{stats_path.stem}_tmp_', dir=stats_path.parent) as tmp_dir:
        tmp_file = Path(tmp_dir) / 'activations'
        tmp_acts = PersistentTensorDict(filename=tmp_file, batch_size=[n_samples], mode='w')
        tmp_acts['acts'] = MMT.empty(shape=(n_samples, n_features), dtype=torch.float32)
        tmp_acts.close()
        tmp_acts = PersistentTensorDict.from_h5(tmp_file, mode='r+')

        try:
            all_labels = []
            counts = torch.zeros(n_classes, dtype=torch.long)
            sums = torch.zeros(n_classes, n_features, dtype=torch.float64)
            offset = 0
            for batch in tqdm(dl, disable=not verbose, total=ceil(n_samples / bs), desc='DMD fit'):
                inputs = batch['image'].to(device)
                labels = batch['label'].long()
                all_labels.append(labels)

                with torch.no_grad():
                    _ = model(inputs)

                acts = model._acts['in_activations'][layer]
                acts = acts.view(acts.shape[0], -1).detach().cpu().float()
                n = acts.shape[0]
                tmp_acts['acts'][offset:offset + n] = acts
                offset += n

                sums.index_add_(0, labels, acts.double())
                counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.long))

            all_labels = torch.cat(all_labels)

            means = torch.zeros(n_classes, n_features)
            non_empty = counts > 0
            means[non_empty] = (sums[non_empty] / counts[non_empty].unsqueeze(1)).float()

            covariance_matrix = torch.zeros(n_features, n_features, dtype=torch.float64)
            cov_dl = DataLoader(tmp_acts, batch_size=bs, shuffle=False,
                                collate_fn=lambda x: x, num_workers=0)
            for i, batch in enumerate(tqdm(cov_dl, disable=not verbose, desc='DMD covariance')):
                acts = batch['acts'].double()
                labels = all_labels[i * bs: i * bs + acts.shape[0]]
                centered = acts - means[labels].double()
                covariance_matrix.addmm_(centered.t(), centered)

            covariance_matrix /= n_samples
            try:
                precision = torch.linalg.pinv(covariance_matrix, hermitian=True).float()
            except TypeError:
                precision = torch.linalg.pinv(covariance_matrix).float()
        finally:
            tmp_acts.close()

    torch.save({'means': means, 'precision': precision}, stats_path)

    if verbose:
        print(f'DMD statistics saved to {stats_path}')


class DMDBase(Score):
    score_name = 'DMD-B'

    def _compute(self, **kwargs):
        model = kwargs['model']
        datasets = kwargs['datasets']
        loaders = kwargs.get('loaders') or list(datasets._dss.keys())
        stats_path = Path(kwargs['stats_path'])
        bs = kwargs.get('batch_size', 128)
        n_threads = kwargs.get('n_threads', 1)
        score_name = kwargs.get('score_name', self.score_name)
        verbose = kwargs.get('verbose', False)

        device = model.device
        layer = _find_classification_head(model)

        stats = torch.load(stats_path, map_location=device, weights_only=True)
        means = stats['means']
        precision = stats['precision']
        n_classes = means.shape[0]

        model.set_target_modules(target_modules=[layer])
        model.set_activations(save_input=True, save_output=False)
        model._model.eval()

        for ds_key in loaders:
            if self._is_computed(ds_key, score_name):
                if verbose:
                    print(ds_key, score_name, 'already computed, skipping')
                continue
            if verbose:
                print('Computing', score_name, 'for dataset', ds_key)

            dss = datasets._dss[ds_key]
            n_samples = len(dss)
            ds_scores = torch.empty(n_samples)
            dl = DataLoader(dss, batch_size=bs, shuffle=False,
                            collate_fn=lambda x: x, num_workers=n_threads)

            write_ptr = 0
            for batch in tqdm(dl, disable=not verbose, desc=f'DMD-B [{ds_key}]'):
                inputs = batch['image'].to(device)
                with torch.no_grad():
                    _ = model(inputs)
                acts = model._acts['in_activations'][layer]
                acts = acts.view(acts.shape[0], -1)
                n = acts.shape[0]
                gaussian_score = torch.zeros(n, n_classes, device=device)
                for c in range(n_classes):
                    zero_f = acts - means[c]
                    term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                    gaussian_score[:, c] = term_gau
                ds_scores[write_ptr:write_ptr + n] = gaussian_score.max(dim=1)[0].detach().cpu()
                write_ptr += n

            self._record(ds_key, ds_scores, score_name)

        return self._df

class DMDPlus(Score):
    score_name = 'dmd_plus'

    def __call__(self, **kwargs):
        id_loader = kwargs.get('id_loader', 'test')
        ood_loaders = kwargs.get('ood_loaders')
        device = kwargs.get('device')
        layer = kwargs.get('layer')
        cvs = kwargs.get('coreavg')
        driller = kwargs.get('driller')
        verbose = kwargs.get('verbose', False)

        all_loaders = [id_loader] + list(ood_loaders)
        pending = self._get_pending_loaders(all_loaders, self.score_name)
        if not pending:
            return self._df

        num_classes = driller.nl_model

        data_ori = cvs._corevds[id_loader][layer].to(device)
        data_ori /= torch.linalg.vector_norm(data_ori, ord=2, dim=1, keepdim=True)
        num_samples = data_ori.shape[0]

        if not self._is_computed(id_loader):
            class_scores = torch.zeros((num_samples, num_classes))
            for c in range(num_classes):
                tensor = data_ori - driller._means[c].view(1, -1)
                class_scores[:, c] = -torch.matmul(
                    torch.matmul(tensor, driller._precision), tensor.t()).diag()
            scores = torch.max(class_scores, dim=1)[0].detach().cpu().reshape(-1)
            self._record(id_loader, scores)

        for ood in ood_loaders:
            if self._is_computed(ood):
                if verbose:
                    print(ood, self.score_name, 'already computed, skipping')
                continue
            data_ood = cvs._corevds[ood][layer].to(device)
            data_ood /= torch.linalg.vector_norm(data_ood, ord=2, dim=1, keepdim=True)
            class_scores = torch.zeros((data_ood.shape[0], num_classes))
            for c in range(num_classes):
                tensor = data_ood - driller._means[c].view(1, -1)
                class_scores[:, c] = -torch.matmul(
                    torch.matmul(tensor, driller._precision), tensor.t()).diag()
            scores = torch.max(class_scores, dim=1)[0].detach().cpu().reshape(-1)
            self._record(ood, scores)

        return self._df


def _dmd_lr_score(train_data, train_label, test_data):
    lr = LogisticRegressionCV(n_jobs=-1, max_iter=5000).fit(train_data, train_label)
    y_train = lr.predict_proba(train_data)[:, 1]
    y_test = lr.predict_proba(test_data)[:, 1]
    return y_train, y_test

def _dmd_lr_score_scaled(train_data, train_label, test_data):
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    lr = LogisticRegressionCV(n_jobs=-1, max_iter=5000).fit(train_data, train_label)
    y_train = lr.predict_proba(train_data)[:, 1]
    y_test = lr.predict_proba(test_data)[:, 1]
    return y_train, y_test

class DMDScore(Score):
    score_name = 'DMD'

    def __call__(self, **kwargs):
        phs = kwargs['peepholes']
        pos_loader_train = kwargs.get('pos_loader_train', 'val')
        pos_loader_test = kwargs.get('pos_loader_test', 'test')
        neg_loaders = kwargs['neg_loaders']
        target_modules = kwargs.get('target_modules') or list(phs._phs[pos_loader_train].keys())
        score_name = kwargs.get('score_name', self.score_name)
        scaling = kwargs.get('scaling', False)

        _score = _dmd_lr_score_scaled if scaling else _dmd_lr_score

        pending_neg = {k: v for k, v in neg_loaders.items()
                       if not self._is_computed(k, f'{score_name}-{k}')}
        if not pending_neg:
            return self._df
        print(f'{score_name}: {len(pending_neg)}/{len(neg_loaders)} neg loaders still to compute: {list(pending_neg.keys())}')
        neg_loaders = pending_neg

        train_pos = torch.stack(
            [phs._phs[pos_loader_train][layer].max(dim=1)[0] for layer in target_modules], dim=1)
        test_pos = torch.stack(
            [phs._phs[pos_loader_test][layer].max(dim=1)[0] for layer in target_modules], dim=1)
        nps = len(train_pos)

        for neg_test_key, neg_train_loaders in neg_loaders.items():
            score_name_neg = f'{score_name}-{neg_test_key}'
            if self._is_computed(neg_test_key, score_name_neg):
                continue
            nnl = len(neg_train_loaders)
            nspnl = floor(nps / nnl)

            train_neg = []
            for nl in neg_train_loaders:
                _train_neg = torch.stack(
                    [phs._phs[nl][layer].max(dim=1)[0] for layer in target_modules], dim=1)
                idx = torch.randperm(len(_train_neg))
                train_neg.append(_train_neg[idx[:nspnl]])
            train_neg = torch.vstack(train_neg)

            train_data = torch.vstack((train_pos, train_neg))
            train_label = torch.hstack(
                (torch.ones(len(train_pos)), torch.zeros(len(train_neg))))
            test_neg = torch.stack(
                [phs._phs[neg_test_key][layer].max(dim=1)[0] for layer in target_modules], dim=1)
            test_data = torch.vstack((test_pos, test_neg))

            _, y_test = _score(train_data, train_label, test_data)

            scores_pos = torch.tensor(y_test)[:len(test_pos)].reshape(-1)
            scores_neg = torch.tensor(y_test)[len(test_pos):].reshape(-1)
            self._record(pos_loader_test, scores_pos, score_name_neg)
            self._record(neg_test_key, scores_neg, score_name_neg)

        return self._df

class DMDScoreConf(Score):
    score_name = 'DMD-in'

    def __call__(self, **kwargs):
        phs = kwargs['peepholes']
        ds = kwargs['dataset']
        loader_train = kwargs.get('loader_train', 'train')
        test_loaders = kwargs.get('test_loaders')
        target_modules = kwargs.get('target_modules') or list(phs._phs[loader_train].keys())
        score_name = kwargs.get('score_name', self.score_name)
        verbose = kwargs.get('verbose', False)
        scaling = kwargs.get('scaling', False)

        _score = _dmd_lr_score_scaled if scaling else _dmd_lr_score

        pending = self._get_pending_loaders(test_loaders, score_name)
        if not pending:
            return self._df
        test_loaders = pending

        idx_neg = torch.argwhere(ds._dss[loader_train]['result'] == 0).squeeze(1)
        idx_pos = torch.argwhere(ds._dss[loader_train]['result'] == 1).squeeze(1)
        perm = torch.randperm(len(idx_pos))[:len(idx_neg)]
        idx_pos = idx_pos[perm]

        train_pos = torch.stack(
            [phs._phs[loader_train][layer].max(dim=1)[0][idx_pos]
             for layer in target_modules], dim=1)
        train_neg = torch.stack(
            [phs._phs[loader_train][layer].max(dim=1)[0][idx_neg]
             for layer in target_modules], dim=1)

        train_data = torch.vstack((train_pos, train_neg))
        train_label = torch.hstack(
            (torch.ones(len(train_pos)), torch.zeros(len(train_neg))))

        for loader_test in test_loaders:
            if self._is_computed(loader_test, score_name):
                if verbose:
                    print(loader_test, score_name, 'already computed, skipping')
                continue
            test_data = torch.stack(
                [phs._phs[loader_test][layer].max(dim=1)[0]
                 for layer in target_modules], dim=1)
            _, y_test = _score(train_data, train_label, test_data)
            scores = torch.tensor(y_test).reshape(-1)
            self._record(loader_test, scores, score_name)

        return self._df
