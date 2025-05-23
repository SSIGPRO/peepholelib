# torch stuff
import torch
from torch.nn.functional import softmax

# python stuff
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
from functools import partial

def get_conceptogram(**kwargs):
    """
    Generate a detailed conceptogram (with network output) for a specific sample.

    Args:
        portion (str): portion of dataset, usually train, test, or eval
        sample (int): Index of the sample to visualize.
        peepholes (Peepholes): Peepholes (already initialized).
        ds (Dataset): Dataset
        corevecs (CoreVectors): Core vectors 
        ph_config_names (list): List of peephole configuration names.
        target_layers (list): List of target layers.
        n_classes (int): Number of classes in the dataset.
        save_path (str): Path to save the generated conceptogram plot.
    """
    path = kwargs['path']
    name = kwargs['name']
    cvs = kwargs['corevectors'] 
    phs = kwargs['peepholes'] 
    portion = kwargs['portion']
    samples = kwargs['samples']
    target_layers = kwargs['target_layers']
    classes = kwargs['classes'] 
    ticks = kwargs['ticks']
    krows = kwargs['krows'] if 'krows' in kwargs else 3
    label_key = kwargs['label_key'] if 'label_key' in kwargs else 'label' 
    pred_fn = kwargs['pred_fn'] if 'pred_fn' in kwargs else partial(softmax, dim=0)

    if len(target_layers) != len(ticks):
        raise ValueError('Number of target layers and ticks should be equal')

    # getting data from corevectors
    _dss = cvs._dss[portion][samples] 
    _phs = phs._phs[portion][samples]
    
    conceptos = []
    for _ph in _phs:
        _c = torch.stack([_ph[layer]['peepholes'] for layer in target_layers])
        conceptos.append(_c)

    path.mkdir(parents=True, exist_ok=True)
    for _d, _c, sample in zip(_dss, conceptos, samples):
        label = int(_d[label_key])
        pred = int(_d['pred'])
        output = pred_fn(_d['output'])
        conf = output.max() 

        _, idx_topk = torch.topk(_c.sum(dim=0), krows,sorted=False)
        classes_topk = [classes[i] for i in idx_topk.tolist()]
        tick_positions = idx_topk.cpu().tolist()
        print(tick_positions)

        tick_labels = [f'{i+1}°: {cls} ({cls_pos})' for i, (cls, cls_pos) in enumerate(zip(classes_topk, tick_positions))]
        print(tick_labels)

        fig = plt.figure(figsize=(5,20))
        gs = gridspec.GridSpec(2, 1, height_ratios=[0.5,3], wspace=0.5, hspace=0.1, figure=fig)
        gss = gs[1].subgridspec(1, 2)
        gs.tight_layout(fig, pad=1)
        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gss[0,:-1]), fig.add_subplot(gss[0,-1])]

        # Plot the image
        axs[0].imshow(_d['image'].permute(1,2,0))
        axs[0].axis('off')
        axs[0].set_title(f'True label: {classes[label]}', fontweight='bold')

        # Plot the conceptogram
        axs[1].imshow(_c.T, aspect=1.2, vmin=0.0, vmax=1.0)
        axs[1].set_xticks(ticks=range(len(ticks)), labels=ticks, rotation=90, fontsize=8)
        axs[1].set_yticks(tick_positions, tick_labels)
        axs[1].yaxis.tick_right()
        axs[1].set_xlabel('Layers')

        # Plot the bar with nn's sofmaxed output
        axs[2].imshow(output.reshape(-1,1), vmin=0.0, vmax=1.0)
        #axs[2].set_title(f'Pred label: {classes[pred]}')
        axs[2].set_xticks([])
        axs[2].set_yticks([pred])
        axs[2].set_yticklabels([f'{classes[pred]} {conf*100:.2f}%'], fontweight='bold')
        axs[2].yaxis.set_label_position("right")
        axs[2].yaxis.tick_right()
        axs[2].set_xlabel('Output')
        
        # save conceptogram
        plt.savefig(path/f'{name}.{portion}.{sample}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Conceptogram saved to {path}")
    return

