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
        portion (str): portion of dataset, usually train, test, or eval
        sample (int): Index of the sample to visualize.
        peepholes (Peepholes): Peepholes (already initialized).
        ds (Dataset): Dataset
        corevecs (CoreVectors): Core vectors 
        ph_config_names (list): List of peephole configuration names.
        target_modules (list): List of modules to include in the conceptogram.
        n_classes (int): Number of classes in the dataset.
        save_path (str): Path to save the generated conceptogram plot.
    """
    path = kwargs.get('path')
    name = kwargs.get('name')
    cvs = kwargs.get('corevectors') 
    phs = kwargs.get('peepholes') 
    loaders = kwargs.get('loaders')
    samples = kwargs.get('samples')
    target_modules = kwargs.get('target_modules')
    pred_fn = kwargs.get('pred_fn', partial(softmax, dim=0))
    label_key = kwargs.get('label_key', 'label')
    protoclasses = kwargs.get('protoclasses', None) 

    # plot text related
    classes = kwargs.get('classes') 
    alt_score = kwargs.get('alt_score', None) 
    alt_score_name = kwargs.get('alt_score_name', 'alt_score')
    ticks = kwargs.get('ticks', target_modules)
    krows = kwargs.get('krows', 3)
    verbose = kwargs.get('verbose', False) 

    if len(target_modules) != len(ticks):
        raise ValueError('Number of target layers and ticks should be equal')
    for portion in loaders:
        # getting data from corevectors
        _dss = cvs._dss[portion][samples] 
        
        conceptos = phs.get_conceptograms(loaders=[portion], target_modules=target_modules)[portion][samples]
        
        #parse alt_scores into a dict for easy access
        if alt_score != None:
            _alt_score = {_sample:_score for _sample, _score in zip(samples, alt_score[portion][samples])}

        path.mkdir(parents=True, exist_ok=True)
        for _d, _c, sample in zip(_dss, conceptos, samples):

            label = int(_d[label_key])
            pred = int(_d['pred'])
            output = pred_fn(_d['output'].squeeze(dim=0))
            conf = output.max() 

            if protoclasses == None:
                fig = plt.figure(figsize=(5 ,20))
                gs = gridspec.GridSpec(2, 1, height_ratios=[0.5,3], wspace=0.5, hspace=0.1, figure=fig)
                gss = gs[1].subgridspec(1, 2)
                gs.tight_layout(fig, pad=1)
                axs = [
                        fig.add_subplot(gs[0, 0]),
                        fig.add_subplot(gss[0,0]),
                        fig.add_subplot(gss[0,1]),
                        ]
            else: 
                fig = plt.figure(figsize=(11 ,20))
                gs = gridspec.GridSpec(2, 1, height_ratios=[0.5,3], wspace=0.5, hspace=0.1, figure=fig)
                gss = gs[1].subgridspec(1, 3)
                gs.tight_layout(fig, pad=1)
                axs = [
                        fig.add_subplot(gs[0, 0]),
                        fig.add_subplot(gss[0,0]),
                        fig.add_subplot(gss[0,1]),
                        fig.add_subplot(gss[0,2])
                        ]

            # Plot the image
            axs[0].imshow(_d['image'].squeeze(dim=0).permute(1,2,0))
            axs[0].axis('off')
            
            if alt_score == None:
                axs[0].set_title(f'True label: {classes[label]}', fontweight='bold')
            else:
                axs[0].set_title(f'True label: {classes[label]} - {alt_score_name}: {_alt_score[sample]:.2f}', fontweight='bold')

            # Plot the protoclasses 
            if not protoclasses == None:
                # add ticks where the protoclasses are high
                _, idx_topk = torch.topk(protoclasses[pred].sum(dim=0), krows, sorted=True)

                classes_topk = [classes[i] for i in idx_topk.tolist()]
                proto_tick_positions = idx_topk.cpu().tolist()
                proto_tick_labels = [f'{i+1}°: {cls} ({cls_pos})' for i, (cls, cls_pos) in enumerate(zip(classes_topk, proto_tick_positions))]

                axs[1].imshow(protoclasses[pred].T, aspect='auto', vmin=0.0, vmax=1.0)
                axs[1].set_xticks(ticks=range(len(ticks)), labels=ticks, rotation=90, fontsize=8)
                axs[1].set_yticks(proto_tick_positions, proto_tick_labels)
                axs[1].set_xlabel('Layers')
                axs[1].set_title('Protoclass')

            # Plot the conceptogram
            _, idx_topk = torch.topk(_c.sum(dim=0), krows, sorted=True)
            classes_topk = [classes[i] for i in idx_topk.tolist()]
            tick_labels = [f'{i+1}°: {cls} ({cls_pos})' for i, (cls, cls_pos) in enumerate(zip(classes_topk, idx_topk))]

            axs[-2].imshow(_c.T, aspect='auto', vmin=0.0, vmax=1.0)
            axs[-2].set_xticks(ticks=range(len(ticks)), labels=ticks, rotation=90, fontsize=8)
            axs[-2].set_yticks(idx_topk, tick_labels)
            axs[-2].yaxis.tick_right()
            axs[-2].set_title('Conceptogram')
            axs[-2].set_xlabel('Layers')

            # Plot the bar with nn's sofmaxed output
            axs[-1].imshow(output.reshape(-1,1), vmin=0.0, vmax=1.0)
            axs[-1].set_xticks([])
            axs[-1].set_yticks([pred])
            axs[-1].set_yticklabels([f'{classes[pred]} {conf*100:.2f}%'], fontweight='bold')
            axs[-1].yaxis.set_label_position("right")
            axs[-1].yaxis.tick_right()
            axs[-1].set_xlabel('Output')
            
            # save conceptogram

            plt.savefig(path/f'{name}.{portion}.{sample}.png', dpi=300, bbox_inches='tight')
            plt.close()
            if verbose: print(f"Conceptogram saved to {path}")
    return

