# torch stuff
import torch
from torch.nn.functional import softmax

# python stuff
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
from functools import partial
from collections import OrderedDict

def plot_conceptogram(**kwargs):
    """
    Plot conceptograms (with network output) for a specific samples.

    Args:
    - path (str): Path to save conceptograms plot.
    - name (str): Name to pre-pend to files.
    - datasets (peepholelib.datasets.parsedDataset.ParsedDataset): parsed dataset.
    - peepholes (peepholelib.peepholes.peepholes): Loaded peepholes, conceptograms are computed by appending the peepholes for several modules.
    - loaders (list[str]): Loaders to take in consideration, usually `['test']`. Defaults to `['test']`.
    - samples (list[int]): List of indexes to visualize plot.
    - target_modules (list[str]): List of target modules to consider to create the conceptograms
    - pref_fn (callable): Prediction function which takes the model's output (`corevectors._dss[<loader>]['output']`) and computes the probability of each class. Defaults to `torch.nn.functional.softmax`.
    - label_key (str): Key to get labels from `corevectors._dss[<loader>][label_key]`. Defaults to `'label'`.
    - protoclasses (torch.tensor): Protoclasses (see `peepholelib.utils.scores.conceptogram_protoclass_score()`) for each label. If given, the conceptograms will include the proroclass respective to the prediction. Defaults to `None`. 
    - verbose (bool): Print progress messages.

    Textual Args:
    - scores (dict(str:dict(str:torch.tensor)))): Scores to add to title(see `peepholelib.utils.scores`) if given. Defaults to `None`.
    - classes (dict({int: str})): Dictionary containing name of the classes given their number.
    - ticks (list[str]): List of modules to put ticks. Defaults to `target_modules`.
    - protoclass_title (str): Title for the protoclass plot.
    - conceptogram_title (str): Title for the conceptogram plot.
    - krows (int): Write the name of `krows` most highlighted classes in the conceptograms.
    """
    path = kwargs['path']
    name = kwargs['name']
    dss = kwargs['datasets']
    phs = kwargs['peepholes'] 
    loaders = kwargs['loaders']
    samples = kwargs['samples']
    target_modules = kwargs['target_modules']
    pred_fn = kwargs.get('pred_fn', partial(softmax, dim=0))
    label_key = kwargs.get('label_key', 'label')
    protoclasses = kwargs.get('protoclasses', None) 
    verbose = kwargs.get('verbose', False) 

    # plot text related
    scores = kwargs.get('scores', None)
    classes = kwargs.get('classes', None) 
    ticks = kwargs.get('ticks', target_modules)
    krows = kwargs.get('krows', 3)
    proto_title = kwargs.get('protoclass_title', 'Protoclass')
    cp_title = kwargs.get('conceptogram_title', 'Conceptogram')

    if len(target_modules) != len(ticks):
        raise ValueError('Number of target layers and ticks should be equal')

    has_title = (scores != None) and (classes != None)

    for ds_key in loaders:
        # getting data from corevectors
        _dss = dss._dss[ds_key][samples] 
        
        conceptos = phs.get_conceptograms(loaders=[ds_key], target_modules=target_modules)[ds_key][samples]
        
        path.mkdir(parents=True, exist_ok=True)
        for _d, _c, sample in zip(_dss, conceptos, samples):

            label = int(_d[label_key])
            pred = int(_d['pred'])
            output = pred_fn(_d['output'].squeeze(dim=0))
            conf = output.max() 

            if protoclasses == None:
                fig = plt.figure(figsize=(5 ,20))
            else: 
                fig = plt.figure(figsize=(11 ,20))

            gs = gridspec.GridSpec(2, 1, height_ratios=[0.5,3], wspace=0.5, hspace=0.1, figure=fig)
            gst = gs[0].subgridspec(1, 1)
            gsb = gs[1].subgridspec(1, 2+int(protoclasses != None))
            gs.tight_layout(fig, pad=1)
            axs = [[fig.add_subplot(axt) for axt in gst], [fig.add_subplot(axb) for axb in gsb]]

            # Plot the image
            axs[0][0].imshow(_d['image'].squeeze(dim=0).permute(1,2,0))
            axs[0][0].axis('off')
            
            if has_title:
                if classes != None: 
                    title = f'True label: {classes[label]}\n'
                else:
                    title = '' 

                if scores != None:
                    for score_name in scores[ds_key]:
                        title += f'\n{score_name}: {scores[ds_key][score_name][sample]:.2f}'

                axs[0][0].axis('off')
                axs[0][0].text(s=title, x=1.0, y=1.0, va='top', transform=axs[0][0].transAxes, fontweight='bold')


            # Plot the protoclasses 
            if not protoclasses == None:
                # add ticks where the protoclasses are high
                _, idx_topk = torch.topk(protoclasses[pred].sum(dim=0), krows, sorted=True)

                classes_topk = [classes[i] for i in idx_topk.tolist()]
                proto_tick_positions = idx_topk.cpu().tolist()
                proto_tick_labels = [f'{i+1}°: {cls} ({cls_pos})' for i, (cls, cls_pos) in enumerate(zip(classes_topk, proto_tick_positions))]

                axs[1][-3].imshow(1-protoclasses[pred].T, aspect='auto', vmin=0.0, vmax=1.0, cmap='bone')
                axs[1][-3].set_xticks(ticks=range(len(ticks)), labels=ticks, rotation=90, fontsize=8)
                axs[1][-3].set_yticks(proto_tick_positions, proto_tick_labels)
                axs[1][-3].set_xlabel('Layers')
                axs[1][-3].set_title(proto_title)

            # Plot the conceptogram
            _, idx_topk = torch.topk(_c.sum(dim=0), krows, sorted=True)
            classes_topk = [classes[i] for i in idx_topk.tolist()]
            tick_labels = [f'{i+1}°: {cls} ({cls_pos})' for i, (cls, cls_pos) in enumerate(zip(classes_topk, idx_topk))]

            axs[1][-2].imshow(1-_c.T, aspect='auto', vmin=0.0, vmax=1.0, cmap='bone')
            axs[1][-2].set_xticks(ticks=range(len(ticks)), labels=ticks, rotation=90, fontsize=8)
            axs[1][-2].set_yticks(idx_topk, tick_labels)
            axs[1][-2].yaxis.tick_right()
            axs[1][-2].set_title(cp_title)
            axs[1][-2].set_xlabel('Layers')

            #Plot the bar with nn's sofmaxed output
            axs[1][-1].imshow(1-output.reshape(-1,1), vmin=0.0, vmax=1.0, cmap='bone')
            axs[1][-1].set_xticks([])
            axs[1][-1].set_yticks([pred])
            axs[1][-1].set_yticklabels([f'{classes[pred]} {conf*100:.2f}%'], fontweight='bold')
            axs[1][-1].yaxis.set_label_position("right")
            axs[1][-1].yaxis.tick_right()
            axs[1][-1].set_xlabel('Output')
            n_layers = len(ticks)
            n_layers = len(ticks)

            base_w = 10.0        # width for small models
            per_layer = 0.22     # extra inches per layer
            max_w = 60.0         # cap to avoid excessive width

            fig = plt.gcf()
            current_w, current_h = fig.get_size_inches()
            fig.set_size_inches(
                min(max_w, base_w + per_layer * n_layers),
                current_h,
                forward=True
            )
            
            # save conceptogram
            plt.savefig(path/f'{name}.{ds_key}.{sample}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            if verbose: print(f"Conceptogram saved to {path}")
    return


def plot_feature_conceptogram(**kwargs):
    """
    Plot feature conceptograms for a specific layer, with X = windows and Y = classes.

    Args:
    - path (Path): Path to save conceptogram plots.
    - name (str): Name to prepend to files.
    - datasets (ParsedDataset): parsed dataset.
    - peepholes (Peepholes): Loaded peepholes object.
    - loaders (list[str]): Loaders to consider, e.g. ['test'].
    - samples (list[int]): Sample indices to visualize.
    - target_modules (list[str]): Base layer names (e.g. ['features.0']).
    - pred_fn (callable): Prediction function on model output. Defaults to softmax over dim=0.
    - label_key (str): Key for labels in dataset entries. Defaults to 'label'.
    - protoclasses (torch.tensor or None): Optional protoclasses tensor.
    - verbose (bool): Print progress messages.

    Text-related Args:
    - scores (dict(str:dict(str:torch.tensor))): Optional scores to show in title.
    - classes (dict[int:str]): Class index -> name.
    - protoclass_title (str): Title for the protoclass plot.
    - conceptogram_title (str): Title for the conceptogram plot.
    - krows (int): Number of top classes to annotate.
    """

    path = kwargs['path']
    name = kwargs['name']
    dss = kwargs['datasets']
    phs = kwargs['peepholes']
    loaders = kwargs['loaders']
    samples = kwargs['samples']

    base_layers = kwargs['target_modules']  # e.g. ['features.0']
    pred_fn = kwargs.get('pred_fn', partial(softmax, dim=0))
    label_key = kwargs.get('label_key', 'label')
    protoclasses = kwargs.get('protoclasses', None)
    verbose = kwargs.get('verbose', False)

    scores = kwargs.get('scores', None)
    classes = kwargs.get('classes', None)
    krows = kwargs.get('krows', 3)
    proto_title = kwargs.get('protoclass_title', 'Protoclass')
    cp_title = kwargs.get('conceptogram_title', 'Conceptogram')

    if classes is None:
        raise ValueError("`classes` dict (int -> str) must be provided for feature conceptograms.")

    for ds_key in loaders:
        _dss = dss._dss[ds_key][samples]

        # ---------------------------------------------------
        # 1) Build window-modules and labels for this loader
        # ---------------------------------------------------
        # Keys present in peephole file for this dataset loader
        all_modules = list(phs._phs[ds_key].keys())

        window_modules = []
        window_labels = []

        for layer in base_layers:
            # All modules for this layer with ".wX" suffix
            layer_windows = [m for m in all_modules if m.startswith(layer + ".w")]

            if len(layer_windows) == 0 and verbose:
                print(f"[plot_feature_conceptogram] No windows found for base layer '{layer}' in '{ds_key}'.")

            # Sort by numeric window index
            def _win_idx(modname):
                digits = []
                cur = ""
                for ch in modname:
                    if ch.isdigit():
                        cur += ch
                    else:
                        if cur:
                            digits.append(int(cur))
                            cur = ""
                if cur:
                    digits.append(int(cur))
                return digits[-1] if digits else 10**9

            layer_windows = sorted(layer_windows, key=_win_idx)
            window_modules.extend(layer_windows)
            window_labels.extend([f"{layer}.w{_win_idx(m)}" for m in layer_windows])

        if len(window_modules) == 0:
            raise ValueError(
                f"No window peepholes found for base layers {base_layers} in dataset '{ds_key}'. "
                f"Available modules: {all_modules}"
            )

        if verbose:
            print(f"[plot_feature_conceptogram] Using window modules for {ds_key}: {window_modules}")

        # ---------------------------------------------------
        # 2) Get conceptograms for these window-modules
        #    conceptos: [n_samples, n_windows, n_classes]
        # ---------------------------------------------------
        conceptos = phs.get_conceptograms(
            loaders=[ds_key],
            target_modules=window_modules
        )[ds_key][samples]

        path.mkdir(parents=True, exist_ok=True)

        # has_title means we will draw text over the image
        has_title = (scores is not None)

        for _d, _c, sample in zip(_dss, conceptos, samples):
            label = int(_d[label_key])
            pred = int(_d['pred'])
            output = pred_fn(_d['output'].squeeze(dim=0))
            conf = output.max()

            # _c: [n_windows, n_classes]
            if _c.dim() != 2:
                raise RuntimeError(
                    f"Conceptogram for sample {sample} expected shape [n_windows, n_classes], got {_c.shape}"
                )

            n_windows, n_classes = _c.shape

            if protoclasses is None:
                fig = plt.figure(figsize=(5, 20))
            else:
                fig = plt.figure(figsize=(11, 20))

            gs = gridspec.GridSpec(
                2, 1, height_ratios=[0.5, 3],
                wspace=0.5, hspace=0.1, figure=fig
            )
            gst = gs[0].subgridspec(1, 1)
            gsb = gs[1].subgridspec(1, 2 + int(protoclasses is not None))
            gs.tight_layout(fig, pad=1)
            axs = [[fig.add_subplot(axt) for axt in gst],
                [fig.add_subplot(axb) for axb in gsb]]

            # -------------------------
            # Top: original image
            # -------------------------
            axs[0][0].imshow(_d['image'].squeeze(dim=0).permute(1, 2, 0))
            axs[0][0].axis('off')

            if has_title:
                title = ""
                title += f"True label: {classes[label]}\n"
                title += f"Pred: {classes[pred]} ({conf*100:.2f}%)\n"
                if scores is not None and ds_key in scores:
                    for score_name in scores[ds_key]:
                        title += f"\n{score_name}: {scores[ds_key][score_name][sample]:.2f}"

                axs[0][0].text(
                    s=title, x=1.0, y=1.0, va='top',
                    transform=axs[0][0].transAxes, fontweight='bold'
                )

            # -------------------------
            # Protoclass heatmap (optional)
            # -------------------------
            if protoclasses is not None:
                # top-k classes from protoclasses[pred] aggregated
                _, idx_topk_proto = torch.topk(
                    protoclasses[pred].sum(dim=0), krows, sorted=True
                )
                classes_topk_proto = [classes[i] for i in idx_topk_proto.tolist()]
                proto_tick_positions = idx_topk_proto.cpu().tolist()
                proto_tick_labels = [
                    f'{i+1}°: {cls} ({cls_pos})'
                    for i, (cls, cls_pos) in enumerate(
                        zip(classes_topk_proto, proto_tick_positions)
                    )
                ]

                axs[1][-3].imshow(
                    1 - protoclasses[pred].T,
                    aspect='auto', vmin=0.0, vmax=1.0, cmap='bone'
                )
                axs[1][-3].set_xticks(
                    ticks=range(len(window_labels)),
                    labels=window_labels,
                    rotation=90,
                    fontsize=8,
                )
                axs[1][-3].set_yticks(proto_tick_positions, proto_tick_labels)
                axs[1][-3].set_xlabel('Windows')
                axs[1][-3].set_title(proto_title)

            # -------------------------
            # Conceptogram heatmap
            # -------------------------
            # Sum over windows -> per-class importance
            _, idx_topk = torch.topk(_c.sum(dim=0), krows, sorted=True)
            classes_topk = [classes[i] for i in idx_topk.tolist()]
            tick_labels = [
                f'{i+1}°: {cls} ({cls_pos})'
                for i, (cls, cls_pos) in enumerate(zip(classes_topk, idx_topk))
            ]

            axs[1][-2].imshow(
                1 - _c.T,  # classes on y, windows on x
                aspect='auto', vmin=0.0, vmax=1.0, cmap='bone'
            )
            axs[1][-2].set_xticks(
                ticks=range(len(window_labels)),
                labels=window_labels,
                rotation=90,
                fontsize=8,
            )
            axs[1][-2].set_yticks(idx_topk, tick_labels)
            axs[1][-2].yaxis.tick_right()
            axs[1][-2].set_title(cp_title)
            axs[1][-2].set_xlabel('Windows')

            # -------------------------
            # Softmax output bar
            # -------------------------
            axs[1][-1].imshow(
                1 - output.reshape(-1, 1),
                vmin=0.0, vmax=1.0, cmap='bone'
            )
            axs[1][-1].set_xticks([])
            axs[1][-1].set_yticks([pred])
            axs[1][-1].set_yticklabels(
                [f'{classes[pred]} {conf*100:.2f}%'],
                fontweight='bold'
            )
            axs[1][-1].yaxis.set_label_position("right")
            axs[1][-1].yaxis.tick_right()
            axs[1][-1].set_xlabel('Output')

            # Dynamic width based on number of windows
            n_layers = len(window_labels)
            base_w = 10.0
            per_layer = 0.22
            max_w = 60.0
            fig = plt.gcf()
            current_w, current_h = fig.get_size_inches()
            fig.set_size_inches(
                min(max_w, base_w + per_layer * n_layers),
                current_h,
                forward=True
            )

            # save conceptogram
            outfile = path / f'{name}.{ds_key}.{sample}.png'
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
            plt.close()

            if verbose:
                print(f"[plot_feature_conceptogram] Saved conceptogram to {outfile}")
        return

