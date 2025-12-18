# torch stuff
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

# python stuff
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec 
import matplotlib.transforms as mtransforms
from functools import partial
from collections import OrderedDict
from peepholelib.datasets.cifar100 import CIFAR100Custom

def _aggregate_output_to_superclasses(output):
    super_output = torch.zeros(20, dtype=output.dtype, device=output.device)
    for coarse_idx, fine_indices in CIFAR100Custom.FINE_TO_COARSE.items():
        for fine_idx in fine_indices:
            super_output[coarse_idx] += output[fine_idx]
    return super_output

def _cap_sorted_order(order, max_rows, descending):
    """
    Function to use when conceptogram has too many rows (concepts): drops the lowest-scoring rows.
    """
    if max_rows is None or order.numel() <= max_rows:
        return order
    return order[:max_rows] if descending else order[-max_rows:]

def _top_concepts_per_layer(conceptogram):
    """
    Return one top concept index per layer.
    Tie-break rule:
    - For layers > 0: prefer the tied concept with the highest value in the previous layer.
    - For the first layer: prefer the tied concept with the highest value in the next layer.
    - If still tied: pick the smallest concept index for deterministic behavior.
    """
    n_layers = conceptogram.shape[0]
    tops = []
    for layer_idx in range(n_layers):
        layer_vals = conceptogram[layer_idx]
        max_val = torch.max(layer_vals)
        candidates = torch.nonzero(layer_vals == max_val, as_tuple=False).flatten()

        if candidates.numel() == 1:
            tops.append(int(candidates.item()))
            continue

        if layer_idx > 0:
            ref_layer = layer_idx - 1
        elif n_layers > 1:
            ref_layer = 1
        else:
            ref_layer = None

        if ref_layer is not None:
            ref_vals = conceptogram[ref_layer, candidates]
            ref_max = torch.max(ref_vals)
            preferred = candidates[ref_vals == ref_max]
        else:
            preferred = candidates

        tops.append(int(torch.min(preferred).item()))
    return tops

def _top_k_concepts_per_layer(conceptogram, top_k):
    """
    Return `top_k` concept paths, where each path is one preferred concept index per layer.

    Path 1 is the usual top concept path. Path r>1 is computed with the same tie-break rule,
    but excluding concepts already selected by earlier paths in that same layer.
    """
    if top_k <= 0:
        return []

    n_layers, n_concepts = conceptogram.shape
    if n_concepts == 0:
        return []

    n_paths = min(int(top_k), n_concepts)
    selected_per_layer = [set() for _ in range(n_layers)]
    paths = []

    for _ in range(n_paths):
        path = []
        for layer_idx in range(n_layers):
            layer_vals = conceptogram[layer_idx]
            available = [
                concept_idx
                for concept_idx in range(n_concepts)
                if concept_idx not in selected_per_layer[layer_idx]
            ]

            if not available:
                path.append(None)
                continue

            candidate_tensor = torch.tensor(available, device=conceptogram.device, dtype=torch.long)
            candidate_vals = layer_vals[candidate_tensor]
            max_val = torch.max(candidate_vals)
            candidates = candidate_tensor[candidate_vals == max_val]

            if candidates.numel() > 1:
                if layer_idx > 0:
                    ref_layer = layer_idx - 1
                elif n_layers > 1:
                    ref_layer = 1
                else:
                    ref_layer = None

                if ref_layer is not None:
                    ref_vals = conceptogram[ref_layer, candidates]
                    ref_max = torch.max(ref_vals)
                    candidates = candidates[ref_vals == ref_max]

            chosen = int(torch.min(candidates).item())
            selected_per_layer[layer_idx].add(chosen)
            path.append(chosen)

        paths.append(path)

    return paths

def _flatten_layer_group(group_spec):
    if isinstance(group_spec, dict):
        flattened = []
        for child in group_spec.values():
            flattened.extend(_flatten_layer_group(child))
        return flattened

    if isinstance(group_spec, (list, tuple)):
        flattened = []
        for child in group_spec:
            if isinstance(child, (dict, list, tuple)):
                flattened.extend(_flatten_layer_group(child))
            else:
                flattened.append(child)
        return flattened

    return [group_spec]

def _group_color_key(name):
    parts = str(name).split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return ' '.join(parts).strip().lower()

def _normalize_target_modules(target_spec):
    """
    Accept either:
    - a flat list/tuple of module names, or
    - an ordered nested mapping describing layer groups.

    Returns:
    - flat list[str] of module names
    - hierarchical group metadata with original contiguous spans
    """
    if isinstance(target_spec, dict):
        group_colors = [
            '#d7ebff',
            '#ffe6d5',
            '#e0f4df',
            '#f9e0ef',
            '#f7efc6',
            '#e5e0fb',
            '#dff3f2',
            '#f4e1d2',
        ]
        flat_modules = []
        groups = []
        color_by_group_key = {}

        def _walk_group(name, value, level, color):
            start = len(flat_modules)

            if isinstance(value, dict):
                for child_name, child_value in value.items():
                    _walk_group(child_name, child_value, level + 1, color)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    if isinstance(child, (dict, list, tuple)):
                        _walk_group(str(name), child, level + 1, color)
                    else:
                        flat_modules.append(child)
            else:
                flat_modules.append(value)

            end = len(flat_modules) - 1
            if end >= start:
                groups.append({
                    'name': str(name),
                    'start': start,
                    'end': end,
                    'level': level,
                    'color': color,
                })

        for group_name, group_value in target_spec.items():
            color_key = _group_color_key(group_name)
            if color_key not in color_by_group_key:
                color_by_group_key[color_key] = group_colors[len(color_by_group_key) % len(group_colors)]
            _walk_group(group_name, group_value, 0, color_by_group_key[color_key])

        return flat_modules, groups

    if isinstance(target_spec, (list, tuple)):
        return list(target_spec), []

    raise TypeError('`target_modules`/`target_layers` must be a list/tuple or a nested dict of groups')

def _build_grouped_display_matrix(matrix, groups, gap_width=1):
    if not groups:
        base_positions = list(range(matrix.shape[1]))
        return matrix, base_positions, [], {idx: idx for idx in base_positions}

    top_groups = [group for group in groups if group['level'] == 0]
    if not top_groups:
        base_positions = list(range(matrix.shape[1]))
        return matrix, base_positions, [], {idx: idx for idx in base_positions}

    blocks = []
    x_positions = []
    display_groups = []
    cursor = 0
    display_column_map = {}

    for group_idx, group in enumerate(top_groups):
        group_block = matrix[:, group['start']:group['end'] + 1]
        blocks.append(group_block)

        display_start = cursor
        display_end = cursor + group_block.shape[1] - 1
        x_positions.extend(range(display_start, display_end + 1))
        for offset, original_col in enumerate(range(group['start'], group['end'] + 1)):
            display_column_map[original_col] = display_start + offset
        cursor = display_end + 1

        if group_idx < len(top_groups) - 1 and gap_width > 0:
            gap = matrix.new_full((matrix.shape[0], gap_width), float('nan'))
            blocks.append(gap)
            cursor += gap_width

    for group in groups:
        display_groups.append({
            **group,
            'display_start': display_column_map[group['start']],
            'display_end': display_column_map[group['end']],
        })

    return torch.cat(blocks, dim=1), x_positions, display_groups, display_column_map

def _lighten_color(color, amount=0.35):
    r, g, b = mcolors.to_rgb(color)
    return (
        1 - (1 - r) * (1 - amount),
        1 - (1 - g) * (1 - amount),
        1 - (1 - b) * (1 - amount),
    )

def _darken_color(color, amount=0.12):
    r, g, b = mcolors.to_rgb(color)
    return (
        r * (1 - amount),
        g * (1 - amount),
        b * (1 - amount),
    )

def _top_level_groups(groups):
    return [group for group in groups if group['level'] == 0]

def _format_group_label(name):
    parts = str(name).split()
    if len(parts) <= 1:
        return str(name)
    return parts[0] + '\n' + ' '.join(parts[1:])

def _style_grouped_ticks(ax, ticklabels, x_positions, display_groups):
    if not ticklabels:
        return

    for label in ax.get_xticklabels():
        label.set_va('top')
        label.set_ha('center')

def _path_color_sequence(top_k, base_color):
    if top_k <= 0:
        return []

    gradient_colors = [
        '#b45309',  # dark amber
        '#ea580c',  # vivid orange
        '#f59e0b',  # amber
        '#facc15',  # yellow
        '#a3e635',  # lime
        '#4ade80',  # green
        '#67e8f9',  # cyan
    ]

    if top_k <= len(gradient_colors):
        return gradient_colors[:top_k]

    colors = list(gradient_colors)
    last_color = mcolors.to_rgb(gradient_colors[-1])
    for extra_idx in range(top_k - len(gradient_colors)):
        amount = min(0.92, 0.12 * (extra_idx + 1))
        colors.append(mcolors.to_hex(_lighten_color(last_color, amount=amount)))
    return colors

def _path_marker_sizes(top_k, largest_size=70, size_step=8, minimum_size=32):
    if top_k <= 0:
        return []

    return [max(minimum_size, largest_size - size_step * idx) for idx in range(top_k)]

def _ordered_unique_concepts_from_paths(paths):
    ordered = []
    seen = set()
    for path in paths:
        for concept_idx in path:
            if concept_idx is None or concept_idx in seen:
                continue
            seen.add(int(concept_idx))
            ordered.append(int(concept_idx))
    return ordered

def _draw_group_hierarchy_bottom(ax, display_groups):
    if not display_groups:
        return

    levels = sorted({group['level'] for group in display_groups}, reverse=True)
    transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    top_offset = -0.125
    level_gap = 0.055

    for idx, level in enumerate(levels):
        y_center = top_offset - idx * level_gap
        level_groups = [group for group in display_groups if group['level'] == level]
        for group in level_groups:
            text_color = 'black'
            group_label = group['name']
            if level == 0:
                box_height = 0.05
                box_y = y_center - box_height / 2
                label_box = mpatches.Rectangle(
                    (group['display_start'] - 0.5, box_y),
                    group['display_end'] - group['display_start'] + 1,
                    box_height,
                    transform=transform,
                    facecolor=group['color'],
                    edgecolor=_darken_color(group['color'], amount=0.18),
                    linewidth=1.0,
                    clip_on=False,
                    zorder=6,
                )
                ax.add_patch(label_box)
                text_color = _darken_color(group['color'], amount=0.55)
                group_label = _format_group_label(group['name'])
            ax.text(
                (group['display_start'] + group['display_end']) / 2,
                y_center,
                group_label,
                transform=transform,
                ha='center',
                va='center',
                fontsize=11,
                fontweight='semibold' if level == 0 else 'normal',
                color=text_color,
                clip_on=False,
                zorder=7,
            )

    for group in [g for g in display_groups if g['level'] > 0]:
        ax.axvline(
            group['display_end'] + 0.5,
            color=group['color'],
            alpha=0.12,
            linewidth=0.8,
            zorder=3,
        )

def _draw_top_group_backgrounds(ax, display_groups):
    for group in _top_level_groups(display_groups):
        ax.axvspan(
            group['display_start'] - 0.5,
            group['display_end'] + 0.5,
            color=group['color'],
            alpha=0.07,
            zorder=2,
        )

def _draw_group_borders(ax, display_groups, n_rows):
    transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for group in _top_level_groups(display_groups):
        rect = mpatches.Rectangle(
            (group['display_start'] - 0.5, 0.0),
            group['display_end'] - group['display_start'] + 1,
            1.0,
            transform=transform,
            fill=False,
            edgecolor='black',
            linewidth=1,
            clip_on=False,
            zorder=6,
        )
        ax.add_patch(rect)

def _style_grouped_heatmap_axes(ax):
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

def _grouped_xlabel_pad(display_groups):
    if not display_groups:
        return None
    n_levels = len({group['level'] for group in display_groups})
    return 88 + max(0, n_levels - 1) * 16

def _top_group_index_by_layer(groups, n_layers):
    mapping = {}
    top_groups = _top_level_groups(groups)
    for group_idx, group in enumerate(top_groups):
        for layer_idx in range(group['start'], group['end'] + 1):
            mapping[layer_idx] = group_idx
    return [mapping.get(layer_idx) for layer_idx in range(n_layers)]

def _get_grouped_cmap(cmap_name, bad_color='#ffffff'):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=bad_color)
    return cmap

def _figure_width_for_layers(n_layers, display_groups, gap_width, has_protoclasses):
    top_group_count = max(1, len(_top_level_groups(display_groups)))
    displayed_columns = n_layers + max(0, top_group_count - 1) * gap_width
    heatmap_width = max(3.0, 0.2 * displayed_columns)
    extra_width = 4.0 if has_protoclasses else 1.8
    return heatmap_width + extra_width

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
    - target_modules (list[str]): Flat list of target modules to consider to create the conceptograms.
    - target_layers (dict | list[str], optional): Grouped layer specification. If provided, only
      `plot_conceptogram2()` uses it to create grouped x-axis sections while internally flattening
      the layers to compute the conceptogram.
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
    - krows (int): Write the name of `krows` most highlighted classes in the protoclass panel.
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
    concepts = kwargs.get('concepts', False)
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
        conceptos = phs.get_conceptograms(loaders=[ds_key], target_modules=target_modules)[ds_key][samples]
        
        path.mkdir(parents=True, exist_ok=True)
        for _c, sample in zip(conceptos, samples):
            _d = dss._dss[ds_key][sample]
            label = int(_d[label_key])
            output = pred_fn(_d['output'].squeeze(dim=0))
            if label_key == 'coarse_label':
                display_output = _aggregate_output_to_superclasses(output)
            else:
                display_output = output
            pred = int(display_output.argmax())
            conf = display_output[pred]

            if protoclasses == None:
                fig = plt.figure(figsize=(12 ,20))
            else: 
                fig = plt.figure(figsize=(17 ,20))

            gs = gridspec.GridSpec(2, 1, height_ratios=[0.5,3], wspace=0.5, hspace=0.1, figure=fig)
            gst = gs[0].subgridspec(1, 1)
            if protoclasses is None:
                gsb = gs[1].subgridspec(1, 1, width_ratios=[3.0])
            else:
                gsb = gs[1].subgridspec(1, 2, width_ratios=[1.0, 3.0])
            gs.tight_layout(fig, pad=1)
            axs = [[fig.add_subplot(axt) for axt in gst], [fig.add_subplot(axb) for axb in gsb]]
            if protoclasses is None:
                concept_ax = axs[1][0]
            else:
                proto_ax = axs[1][0]
                concept_ax = axs[1][1]

            # Plot the image
            axs[0][0].imshow(_d['image'].squeeze(dim=0).permute(1,2,0))
            axs[0][0].axis('off')
            
            if has_title:
                if classes != None: 
                    title = f'True label: {classes.get(int(label), str(int(label)))}\n'
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

                classes_topk = [classes.get(int(i), str(int(i))) for i in idx_topk.tolist()]
                proto_tick_positions = idx_topk.cpu().tolist()
                proto_tick_labels = [f'{i+1}°: {cls} ({cls_pos})' for i, (cls, cls_pos) in enumerate(zip(classes_topk, proto_tick_positions))]

                proto_im = proto_ax.imshow(1-protoclasses[pred].T, aspect='auto', vmin=0.0, vmax=1.0, cmap='bone')
                proto_ax.set_xticks(ticks=range(len(ticks)), labels=ticks, rotation=90, fontsize=8)
                proto_ax.set_yticks(proto_tick_positions, proto_tick_labels)
                proto_ax.set_xlabel('Layers')
                proto_ax.set_title(proto_title)

            # Plot the conceptogram
            _, idx_topk = torch.topk(_c.sum(dim=0), krows, sorted=True)
           
            classes_topk = [classes.get(int(i), str(int(i))) for i in idx_topk.tolist()]
            tick_labels = [f'{i+1}°: {cls} ({cls_pos})' for i, (cls, cls_pos) in enumerate(zip(classes_topk, idx_topk))]

            concept_im = concept_ax.imshow(1-_c.T, aspect='auto', cmap='bone')
            concept_ax.set_xticks(ticks=range(len(ticks)), labels=ticks, rotation=90, fontsize=8)
            concept_ax.set_yticks(idx_topk, tick_labels)
            concept_ax.yaxis.tick_right()
            concept_ax.set_title(cp_title)
            concept_ax.set_xlabel('Layers')

            # Softmax output bar intentionally removed.
            
            # save conceptogram
            plt.savefig(path/f'{name}.{ds_key}.{sample}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            if verbose: print(f"Conceptogram saved to {path}")
    return

def plot_conceptogram2(**kwargs):
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
    - krows (int): Write the name of `krows` most highlighted classes in the protoclass panel.

    Extra:
    - concepts (bool): If True, use dataset concept keys for labels (instead of `classes`).
    - cmap (str): Matplotlib colormap to use for heatmaps. Defaults to 'viridis_r' (high=dark).
    - reverse_order (bool): If True, puts highest concepts at the TOP. Defaults to False
                            (highest at the BOTTOM, as you requested).
    - colorbar_label (str): Label shown in colorbars. Defaults to 'Activation'.
    - max_rows (int): Maximum number of displayed rows in heatmaps. Defaults to 100.
    - highlight_top_path (bool): If True, mark top concept(s) per layer and connect
                                 them with inter-layer path segments. Defaults to True.
    - top_path_color (str): Color used for top markers and path. Defaults to 'orange'.
    - top_path_alpha (float): Alpha used for path overlay. Defaults to 0.9.
    - top_path_linewidth (float): Line width used for path segments. Defaults to 1.8.
    - group_gap (int): Number of blank columns inserted between groups. Defaults to 1.
    """
    path = kwargs['path']
    name = kwargs['name']
    dss = kwargs['datasets']
    phs = kwargs['peepholes']
    loaders = kwargs['loaders']
    samples = kwargs['samples']
    target_spec = kwargs['target_layers'] if 'target_layers' in kwargs else kwargs['target_modules']
    target_modules, layer_groups = _normalize_target_modules(target_spec)
    pred_fn = kwargs.get('pred_fn', partial(softmax, dim=0))
    label_key = kwargs.get('label_key', 'label')
    concepts = kwargs.get('concepts', False)
    protoclasses = kwargs.get('protoclasses', None)
    verbose = kwargs.get('verbose', False)
    colorbar = kwargs.get('colorbar', True)

    # plot text related
    scores = kwargs.get('scores', None)
    classes = kwargs.get('classes', None)
    ticks = kwargs.get('ticks', target_modules)
    krows = kwargs.get('krows', 3)
    proto_title = kwargs.get('protoclass_title', 'Protoclass')
    max_rows = kwargs.get('max_rows', 100)
    highlight_top_path = kwargs.get('highlight_top_path', True)
    top_path_color = kwargs.get('top_path_color', 'orange')
    top_path_alpha = kwargs.get('top_path_alpha', 0.9)
    top_path_linewidth = kwargs.get('top_path_linewidth', 2)
    group_gap = kwargs.get('group_gap', 1)

    cmap = kwargs.get('cmap', 'bone_r')  
    reverse_order = kwargs.get('reverse_order', False)  # False -> low at top, high at bottom
    grouped_cmap = _get_grouped_cmap(cmap)
    top_group_by_layer = _top_group_index_by_layer(layer_groups, len(target_modules))

    if len(target_modules) != len(ticks):
        raise ValueError('Number of target layers and ticks should be equal')

    for ds_key in loaders:
        # getting data from corevectors
        _dss = dss._dss[ds_key][samples]

        concept_keys = None
        if concepts:
            sample0 = _dss[0]
            concept_keys = [k for k in sample0.keys() if k not in ('image', 'label', 'bbox', 'output', 'result', 'pred')]
            concept_keys = sorted(concept_keys)

        conceptos = phs.get_conceptograms(loaders=[ds_key], target_modules=target_modules)[ds_key][samples]

        path.mkdir(parents=True, exist_ok=True)
        for _d, _c, sample in zip(_dss, conceptos, samples):

            label = int(_d[label_key])
            output = pred_fn(_d['output'].squeeze(dim=0))
            if label_key == 'coarse_label':
                display_output = _aggregate_output_to_superclasses(output)
            else:
                display_output = output
            pred = int(display_output.argmax())
            conf = display_output[pred]

            fig_width = _figure_width_for_layers(
                n_layers=len(target_modules),
                display_groups=layer_groups,
                gap_width=group_gap,
                has_protoclasses=protoclasses is not None,
            )
            fig = plt.figure(figsize=(fig_width, 20))

            gs = gridspec.GridSpec(2, 1, height_ratios=[0.5, 3], wspace=0.5, hspace=0.1, figure=fig)
            gst = gs[0].subgridspec(1, 1)
            if protoclasses is None:
                gsb = gs[1].subgridspec(1, 1, width_ratios=[3.0])
            else:
                gsb = gs[1].subgridspec(1, 2, width_ratios=[1.0, 3.0])
            gs.tight_layout(fig, pad=1)
            axs = [[fig.add_subplot(axt) for axt in gst], [fig.add_subplot(axb) for axb in gsb]]

            # Plot the image
            axs[0][0].imshow(_d['image'].squeeze(dim=0).permute(1, 2, 0))
            axs[0][0].axis('off')

            if classes is None:
                true_label = str(label)
                pred_label = str(pred)
            else:
                true_label = classes.get(int(label), str(int(label)))
                pred_label = classes.get(int(pred), str(int(pred)))

            title = f'True label: {true_label}\nPredicted: {pred_label} ({conf*100:.2f}%)'

            if scores is not None:
                for score_name in scores[ds_key]:
                    title += f'\n{score_name}: {scores[ds_key][score_name][sample]:.2f}'

            axs[0][0].axis('off')
            axs[0][0].set_title(
                title,
                fontsize=15,
                fontweight='semibold',
                fontfamily='DejaVu Sans',
                loc='center',
                pad=12
            )

            if protoclasses is None:
                concept_ax = axs[1][0]
            else:
                proto_ax = axs[1][0]
                concept_ax = axs[1][1]

            # -------------------------
            # Plot the protoclasses (ordered by magnitude, layers unchanged)
            # -------------------------
            if protoclasses is not None:
                proto = protoclasses[pred]  # expected shape: (n_layers, n_classes_or_concepts)
                proto_score = proto.sum(dim=0)

                # order: low at top, high at bottom (unless reverse_order=True)
                proto_order = torch.argsort(proto_score, descending=reverse_order)
                proto_order = _cap_sorted_order(proto_order, max_rows=max_rows, descending=reverse_order)
                proto_sorted = proto[:, proto_order]

                proto_k = min(krows, proto_order.numel())
                if reverse_order:
                    proto_topk = proto_order[:proto_k]
                    proto_topk_positions = list(range(proto_k))
                else:
                    proto_topk = proto_order[-proto_k:]
                    proto_topk_positions = list(range(proto_order.numel() - proto_k, proto_order.numel()))

                if classes is None:
                    proto_tick_labels = [str(i) for i in proto_topk.tolist()]
                else:
                    proto_classes_topk = [classes.get(int(i), str(int(i))) for i in proto_topk.tolist()]
                    proto_tick_labels = [
                        f'{rank+1}°: {cls} ({orig_idx})'
                        for rank, (cls, orig_idx) in enumerate(zip(proto_classes_topk, proto_topk.tolist()))
                    ]

                proto_matrix = proto_sorted.T
                proto_display, proto_x_positions, proto_groups, _ = _build_grouped_display_matrix(
                    proto_matrix,
                    layer_groups,
                    gap_width=group_gap,
                )
                proto_im = proto_ax.imshow(proto_display, aspect='auto', vmin=0.0, vmax=1.0, cmap=grouped_cmap)
                proto_ax.set_xticks(ticks=proto_x_positions, labels=ticks, rotation=90, fontsize=9)
                proto_ax.tick_params(axis='x', pad=8)
                proto_ax.set_yticks(proto_topk_positions, proto_tick_labels)
                proto_ax.set_xlabel('Layers', labelpad=_grouped_xlabel_pad(proto_groups))
                proto_ax.set_title(proto_title)
                _draw_top_group_backgrounds(proto_ax, proto_groups)
                _draw_group_borders(proto_ax, proto_groups, proto_display.shape[0])
                _style_grouped_heatmap_axes(proto_ax)
                _style_grouped_ticks(proto_ax, ticks, proto_x_positions, proto_groups)
                _draw_group_hierarchy_bottom(proto_ax, proto_groups)
                if colorbar:
                    fig.colorbar(proto_im, ax=proto_ax, fraction=0.028, pad=0.04, location='left')

            # -------------------------
            # Plot the conceptogram (ordered by magnitude, layers unchanged)
            # -------------------------
            # _c is expected shape: (n_layers, n_concepts) given your original _c.T usage
            top_by_layer = _top_concepts_per_layer(_c)

            concept_score = _c.sum(dim=0)
            # order: low at top, high at bottom (unless reverse_order=True)
            order = torch.argsort(concept_score, descending=reverse_order)
            if max_rows is not None and order.numel() > max_rows:
                capped_order = _cap_sorted_order(order, max_rows=max_rows, descending=reverse_order)
                mandatory = torch.tensor(sorted(set(int(i) for i in top_by_layer)), device=_c.device, dtype=torch.long)
                merged = torch.unique(torch.cat([capped_order, mandatory], dim=0))
                merged_scores = concept_score[merged]
                merged_rank = torch.argsort(merged_scores, descending=reverse_order)
                order = merged[merged_rank]
            _c_sorted = _c[:, order]

            row_of_concept = {int(concept_idx): pos for pos, concept_idx in enumerate(order.tolist())}

            # Show y-ticks for any concept that is top in at least one layer (has a dot).
            last_layer_top1 = int(top_by_layer[-1])
            concepts_with_dots = sorted(
                [idx for idx in set(int(i) for i in top_by_layer) if idx in row_of_concept],
                key=lambda idx: row_of_concept[idx],
            )
            top1_positions = [row_of_concept[idx] for idx in concepts_with_dots]

            if concepts:
                aligned_concept_keys = list(concept_keys)[:_c.shape[1]]
                tick_labels = []
                for orig_idx in concepts_with_dots:
                    concept_name = aligned_concept_keys[orig_idx] if orig_idx < len(aligned_concept_keys) else str(orig_idx)
                    prefix = 'Top1' if orig_idx == last_layer_top1 else ''
                    tick_labels.append(f'{prefix}: {concept_name}')
            else:
                if classes is None:
                    tick_labels = [
                        f'{"Top1" if orig_idx == last_layer_top1 else ""}: {orig_idx}'
                        for orig_idx in concepts_with_dots
                    ]
                else:
                    tick_labels = [
                        f'{"Top1" if orig_idx == last_layer_top1 else ""}: {classes.get(orig_idx, str(orig_idx))}'
                        for orig_idx in concepts_with_dots
                    ]

            concept_matrix = _c_sorted.T
            concept_display, concept_x_positions, concept_groups, concept_column_map = _build_grouped_display_matrix(
                concept_matrix,
                layer_groups,
                gap_width=group_gap,
            )
            concept_im = concept_ax.imshow(concept_display, aspect='auto', vmin=0.0, vmax=1.0, cmap=grouped_cmap)
            concept_ax.set_xticks(ticks=concept_x_positions, labels=ticks, rotation=90, fontsize=9)
            concept_ax.tick_params(axis='x', pad=8)
            concept_ax.set_yticks(top1_positions, tick_labels)
            concept_ax.tick_params(axis='y', labelsize=15)
            concept_ax.yaxis.tick_right()
            concept_ax.set_xlabel('Layers', labelpad=_grouped_xlabel_pad(concept_groups))
            _draw_top_group_backgrounds(concept_ax, concept_groups)
            _draw_group_borders(concept_ax, concept_groups, concept_display.shape[0])
            _style_grouped_heatmap_axes(concept_ax)
            _style_grouped_ticks(concept_ax, ticks, concept_x_positions, concept_groups)
            _draw_group_hierarchy_bottom(concept_ax, concept_groups)

            layer_x_positions = [concept_column_map[idx] for idx in range(len(target_modules))]

            if highlight_top_path:
                # Mark one preferred top concept per layer and connect adjacent layers.
                for layer_idx, concept_idx in enumerate(top_by_layer):
                    y = row_of_concept.get(int(concept_idx))
                    if y is None:
                        continue
                    concept_ax.scatter(
                        layer_x_positions[layer_idx],
                        y,
                        s=45,
                        facecolors=top_path_color,
                        edgecolors=top_path_color,
                        linewidths=1.0,
                        alpha=top_path_alpha,
                        zorder=4,
                    )

                for layer_idx in range(len(top_by_layer) - 1):
                    if top_group_by_layer[layer_idx] != top_group_by_layer[layer_idx + 1]:
                        continue
                    y0 = row_of_concept.get(int(top_by_layer[layer_idx]))
                    y1 = row_of_concept.get(int(top_by_layer[layer_idx + 1]))
                    if y0 is None or y1 is None:
                        continue
                    concept_ax.plot(
                        [layer_x_positions[layer_idx], layer_x_positions[layer_idx + 1]],
                        [y0, y1],
                        color=top_path_color,
                        alpha=top_path_alpha,
                        linewidth=top_path_linewidth,
                        zorder=3,
                    )

            if colorbar:
                fig.colorbar(concept_im, ax=concept_ax, fraction=0.028, pad=0.04, location='left')

            # save conceptogram
            plt.savefig(path / f'{name}.{ds_key}.{sample}.png', dpi=300, bbox_inches='tight')
            plt.close()

            if verbose:
                print(f"Conceptogram saved to {path}")

    return

def plot_conceptogram3(**kwargs):
    """
    Plot only the top-k concept paths for each sample, without the concept heatmap.

    Args are mostly the same as `plot_conceptogram2()`, with these differences:
    - `top_k_concept` (int): Number of concept paths to draw. Defaults to 1.
    - No heatmap-related rendering is performed.
    - Prediction is taken directly from the model output (no superclass aggregation).
    - Top-ranked concept paths use slightly larger markers than lower-ranked ones.
    - Uses a lighter default export path than the heatmap plots to keep save times down.
    """
    path = kwargs['path']
    name = kwargs['name']
    dss = kwargs['datasets']
    phs = kwargs['peepholes']
    loaders = kwargs['loaders']
    samples = kwargs['samples']
    target_spec = kwargs['target_layers'] if 'target_layers' in kwargs else kwargs['target_modules']
    target_modules, layer_groups = _normalize_target_modules(target_spec)
    pred_fn = kwargs.get('pred_fn', partial(softmax, dim=0))
    label_key = kwargs.get('label_key', 'label')
    concepts = kwargs.get('concepts', False)
    verbose = kwargs.get('verbose', False)

    scores = kwargs.get('scores', None)
    classes = kwargs.get('classes', None)
    ticks = kwargs.get('ticks', target_modules)
    top_k_concept = max(1, int(kwargs.get('top_k_concept', 1)))
    top_path_color = kwargs.get('top_path_color', 'orange')
    top_path_alpha = kwargs.get('top_path_alpha', 0.9)
    top_path_linewidth = kwargs.get('top_path_linewidth', 2)
    group_gap = kwargs.get('group_gap', 1)
    reverse_order = kwargs.get('reverse_order', False)
    order_concepts_by_row_sum = kwargs.get('order_concepts_by_row_sum', False)

    if len(target_modules) != len(ticks):
        raise ValueError('Number of target layers and ticks should be equal')

    top_group_by_layer = _top_group_index_by_layer(layer_groups, len(target_modules))

    for ds_key in loaders:
        _dss = dss._dss[ds_key][samples]

        concept_keys = None
        if concepts:
            sample0 = _dss[0]
            concept_keys = [k for k in sample0.keys() if k not in ('image', 'label', 'bbox', 'output', 'result', 'pred')]
            concept_keys = sorted(concept_keys)

        conceptos = phs.get_conceptograms(loaders=[ds_key], target_modules=target_modules)[ds_key][samples]

        path.mkdir(parents=True, exist_ok=True)
        for _d, _c, sample in zip(_dss, conceptos, samples):
            label = int(_d[label_key])
            output = pred_fn(_d['output'].squeeze(dim=0))
            if label_key == 'coarse_label':
                display_output = _aggregate_output_to_superclasses(output)
            else:
                display_output = output
            pred = int(display_output.argmax())
            conf = display_output[pred]

            fig_width = _figure_width_for_layers(
                n_layers=len(target_modules),
                display_groups=layer_groups,
                gap_width=group_gap,
                has_protoclasses=False,
            )
            fig = plt.figure(figsize=(fig_width, 14))

            gs = gridspec.GridSpec(2, 1, height_ratios=[0.5, 3], wspace=0.5, hspace=0.1, figure=fig)
            gst = gs[0].subgridspec(1, 1)
            gsb = gs[1].subgridspec(1, 1, width_ratios=[3.0])
            axs = [[fig.add_subplot(axt) for axt in gst], [fig.add_subplot(axb) for axb in gsb]]
            fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.20, hspace=0.24)

            axs[0][0].imshow(_d['image'].squeeze(dim=0).permute(1, 2, 0))
            axs[0][0].axis('off')

            if classes is None:
                true_label = str(label)
                pred_label = str(pred)
            else:
                true_label = classes.get(int(label), str(int(label)))
                pred_label = classes.get(int(pred), str(int(pred)))

            title = f'True label: {true_label}\nPredicted: {pred_label} ({conf*100:.2f}%)'

            if scores is not None:
                for score_name in scores[ds_key]:
                    title += f'\n{score_name}: {scores[ds_key][score_name][sample]:.2f}'

            axs[0][0].set_title(
                title,
                fontsize=15,
                fontweight='semibold',
                fontfamily='DejaVu Sans',
                loc='center',
                pad=12
            )

            concept_ax = axs[1][0]
            top_paths = _top_k_concepts_per_layer(_c, top_k_concept)
            selected_concepts = _ordered_unique_concepts_from_paths(top_paths)

            if not selected_concepts:
                selected_concepts = []

            if order_concepts_by_row_sum:
                concept_score = _c.sum(dim=0)
                selected_tensor = torch.tensor(selected_concepts, device=_c.device, dtype=torch.long) if selected_concepts else None
                if selected_tensor is not None and selected_tensor.numel() > 0:
                    selected_scores = concept_score[selected_tensor]
                    selected_rank = torch.argsort(selected_scores, descending=reverse_order)
                    ordered_selected = selected_tensor[selected_rank].tolist()
                else:
                    ordered_selected = []
            else:
                ordered_selected = selected_concepts

            row_of_concept = {int(concept_idx): pos for pos, concept_idx in enumerate(ordered_selected)}
            n_rows = max(1, len(ordered_selected))
            layout_matrix = torch.zeros((n_rows, len(target_modules)), dtype=_c.dtype, device=_c.device)
            _, concept_x_positions, concept_groups, concept_column_map = _build_grouped_display_matrix(
                layout_matrix,
                layer_groups,
                gap_width=group_gap,
            )

            if concepts:
                aligned_concept_keys = list(concept_keys)[:_c.shape[1]] if concept_keys is not None else []
                tick_labels = [
                    aligned_concept_keys[idx] if idx < len(aligned_concept_keys) else str(idx)
                    for idx in ordered_selected
                ]
            else:
                if classes is None:
                    tick_labels = [str(idx) for idx in ordered_selected]
                else:
                    tick_labels = [classes.get(idx, str(idx)) for idx in ordered_selected]

            concept_ax.set_xlim(-0.5, max(concept_x_positions) + 0.5 if concept_x_positions else len(target_modules) - 0.5)
            concept_ax.set_ylim(n_rows - 0.5, -0.5)
            concept_ax.set_xticks(ticks=concept_x_positions, labels=ticks, rotation=90, fontsize=9)
            concept_ax.tick_params(axis='x', pad=8)
            concept_ax.set_yticks(list(range(len(ordered_selected))), tick_labels)
            concept_ax.tick_params(axis='y', labelsize=15)
            concept_ax.yaxis.tick_right()
            concept_ax.set_xlabel('Layers', labelpad=_grouped_xlabel_pad(concept_groups))
            _draw_group_borders(concept_ax, concept_groups, n_rows)
            _style_grouped_heatmap_axes(concept_ax)
            _style_grouped_ticks(concept_ax, ticks, concept_x_positions, concept_groups)
            _draw_group_hierarchy_bottom(concept_ax, concept_groups)

            layer_x_positions = [concept_column_map[idx] for idx in range(len(target_modules))]
            path_colors = _path_color_sequence(len(top_paths), top_path_color)
            path_marker_sizes = _path_marker_sizes(len(top_paths))

            if top_paths:
                legend_handles = [
                    mpatches.Patch(facecolor=path_colors[path_idx], edgecolor=path_colors[path_idx], label=f'Top{path_idx + 1} concept')
                    for path_idx in range(len(top_paths))
                ]
                concept_ax.legend(
                    handles=legend_handles,
                    loc='upper left',
                    bbox_to_anchor=(0.01, 0.99),
                    fontsize=12,
                    frameon=True,
                    facecolor='white',
                    edgecolor='#cfcfcf',
                    framealpha=0.98,
                    borderaxespad=0.4,
                    borderpad=0.7,
                    handlelength=1.4,
                    handletextpad=0.7,
                )

            yticklabels = concept_ax.get_yticklabels()
            for path_idx, top_by_layer in enumerate(top_paths):
                if not top_by_layer:
                    continue
                last_layer_concept = top_by_layer[-1]
                if last_layer_concept is None or last_layer_concept not in row_of_concept:
                    continue
                tick_row = row_of_concept[int(last_layer_concept)]
                tick_label = yticklabels[tick_row]
                tick_color = path_colors[path_idx]
                tick_label.set_bbox({
                    'boxstyle': 'square,pad=0.25',
                    'facecolor': tick_color,
                    'edgecolor': _darken_color(tick_color, amount=0.18),
                    'linewidth': 1.5,
                })
                tick_label.set_color(_darken_color(tick_color, amount=0.55))

            for path_idx, top_by_layer in enumerate(top_paths):
                color = path_colors[path_idx]
                marker_size = path_marker_sizes[path_idx]
                for layer_idx, concept_idx in enumerate(top_by_layer):
                    if concept_idx is None:
                        continue
                    y = row_of_concept.get(int(concept_idx))
                    if y is None:
                        continue
                    concept_ax.scatter(
                        layer_x_positions[layer_idx],
                        y,
                        s=marker_size,
                        facecolors=color,
                        edgecolors=color,
                        linewidths=1.0,
                        alpha=top_path_alpha,
                        zorder=4,
                    )

            plt.savefig(path / f'{name}.{ds_key}.{sample}.png', dpi=180)
            plt.close()

            if verbose:
                print(f"Conceptogram saved to {path}")

    return
