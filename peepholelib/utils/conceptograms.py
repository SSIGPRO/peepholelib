# torch stuff
import torch
from tensordict import MemoryMappedTensor as MMT
import matplotlib.gridspec as gridspec


# python stuff
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


verbose = False
splits = ['train', 'test', 'val']


def _extract_outputs(corevecs):
    o_dnn = {}
    o_dnn_dfs = {}

    with corevecs as cv:
        cv.load_only(loaders=splits, verbose=True)

        for split in splits:
            act_data = cv._actds[split]
            print("act data keys", act_data.keys())
            
            # Load softmax outputs 
            outputs = act_data['output'][:]
            
            # Load true labels and results
            true_labels = torch.tensor(act_data['label'][:])
            r = act_data['result'][:].numpy()

            o_dnn[split] = outputs

            # Confidence (max prob) and predicted class
            out_max = np.max(outputs.numpy(), axis=1)
            out_label = np.argmax(outputs, axis=1)

            out_df = pd.DataFrame()
            out_df['max'] = out_max
            out_df['label'] = out_label
            out_df['true'] = true_labels.numpy().astype(int)
            out_df['result'] = r

            o_dnn_dfs[split] = out_df

            if verbose: print(f'{split} → outputs shape: {outputs.shape}, df shape: {out_df.shape}')

    
    return o_dnn, o_dnn_dfs

def _initialize_peepholes(peepholes):
    """
    Loads peephole classifiers for conceptograms.
    
    Args:
        peepholes (Peepholes): peepholes (already initialized).
    
    Returns:
        dict: A dictionary containing loaded peepholes for each configuration and split.
    """
    ph_dict = {}  
        
    with peepholes as ph:
        ph.load_only(loaders=splits, verbose=False)
        for split in splits:
            ph_dict.setdefault(ph.name , {})[split] = ph._phs[split].detach().cpu()
    
    return ph_dict, ph.name
    
def _generate_conceptograms_dict(peepholes, target_layers, n_classes):
    """
    Constructs a tensor for each data split with shape (samples, layers, classes) for Conceptograms.
    
    Args:
        peepholes (Peepholes): Peepholes (already initialized).
        target_layers (list): List of target layers to extract peephole data from.
        n_classes (int): Number of classes in the dataset.
    
    Returns:
        dict: A dictionary containing conceptogram matrices for each split ('train', 'test', 'val').
    """
    
    ph_dict, peephole_config_name = _initialize_peepholes(
        peepholes=peepholes,
    )

    cgs_dict = {}
    
        
    # Get the first layer in the 'train' split as a reference
    layer = list(ph_dict[peephole_config_name]['train'].keys())[0]

    # Allocate memory for each dataset split
    for key in ph_dict[peephole_config_name].keys():
        cgs_dict[key] = MMT.empty(shape=torch.Size(
            (len(ph_dict[peephole_config_name][key][layer]['peepholes']), 
            len(ph_dict[peephole_config_name]['train'].keys()), 
            n_classes)))
        if verbose: print(f"Allocated memory for {key}: {cgs_dict[key].shape}")

    # Populate the tensor dictionary with peephole data
    for key in ph_dict[peephole_config_name].keys():
        if verbose: print(f'\n------------{key}-------------')
        for i, layer in enumerate(target_layers):
            cgs_dict[key][:, i, :] = ph_dict[peephole_config_name][key][layer]['peepholes']
            if verbose: print(f"Sample data for {key}, layer {layer}:\n", cgs_dict[key][0])
    
    return cgs_dict

def generate_conceptograms(peepholes, save_path, n_cgs=5, **kwargs):
    """
    Generate conceptograms for random samples.
    
    Args:
        peepholes (Peepholes): Peepholes (already initialized).
        save_path (str): Directory where conceptograms will be saved.
        n_cgs (int): Number of conceptograms to display (default: 5).
        target_layers (list): List of target layers.
        n_classes (int): Number of classes.
    """
    os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists

    cgs_dict = _generate_conceptograms_dict(
        peepholes=peepholes,
        target_layers=kwargs['target_layers'],
        n_classes=kwargs['n_classes'],
    )

    split = 'test'

    fig, axs = plt.subplots(1, n_cgs, figsize=(10, 6))
    fig.suptitle(f'Examples of conceptograms - split={split}\n')
    for i in range(n_cgs):
        axs[i].imshow(cgs_dict[split][i].detach().cpu().numpy().T, cmap='YlGnBu')
    plt.tight_layout()

    # save conceptograms
    filename = f'conceptograms_{split}_{n_cgs}samples.png'
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file)
    print(f"Conceptograms saved to: {save_file}")
    plt.close()



def generate_conceptogram(peepholes, sample, save_path, **kwargs):
    """
    Generate a single conceptogram (sample specific).
    
    Args:
        peepholes (Peepholes): Peepholes (already initialized).
        sample (int): Index of split test sample to visualize.
        save_path (str): Path where the generated conceptogram should be saved.
        target_layers (list): List of target layers.
        n_classes (int): Number of classes.
    """
    split = 'test'
    
    # Generate conceptogram dictionary
    cgs_dict = _generate_conceptograms_dict(
        peepholes=peepholes,
        target_layers=kwargs['target_layers'],
        n_classes=kwargs['n_classes'],
    )


    # Create conceptogram
    fig, axs = plt.subplots(1, 1, figsize=(3, 6))
    fig.suptitle(f'Example of conceptogram - split={split}, index={sample}\n')
    axs.imshow(cgs_dict[split][sample].detach().cpu().numpy().T, aspect='auto', cmap='YlGnBu')
    
    plt.tight_layout()
    
    # Save conceptogram
    os.makedirs(save_path, exist_ok=True)
    filename = f'conceptogram_{split}_sample{sample}.png'
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file)
    plt.close()
    
    print(f'Conceptogram saved at: {save_file}')



def generate_sample_conceptogram(sample, **kwargs):
    split = 'test'
    save_path = kwargs['save_path']
    ds = kwargs['ds']

    o_dnn, o_dnn_dfs = _extract_outputs(kwargs['corevecs'])

    if sample >= len(o_dnn_dfs[split]):
        raise ValueError(f"Sample must be between 0 and {len(o_dnn_dfs[split]) - 1}! You passed {sample}.")

    cgs_dict = _generate_conceptograms_dict(
        peepholes=kwargs['peepholes'],
        target_layers=kwargs['target_layers'],
        n_classes=kwargs['n_classes'],
    )

    # Labels 
    true_out = o_dnn_dfs[split]['true'][sample]
    true_class = ds._classes[true_out]
    label_out = o_dnn_dfs[split]['label'][sample]
    label_class = ds._classes[label_out]
    confidence = o_dnn_dfs[split]['max'][sample]

    # Data
    concepto = cgs_dict[split][sample].detach().cpu().numpy().T
    out_net = np.expand_dims(o_dnn[split][sample], axis=1)
    image = ds._dss[split][sample][0].permute(1, 2, 0).cpu().numpy()

    # plotting
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 0.5])  

    fig.suptitle(
        f'Split: {split} - Sample: {sample}\nTrue: {true_class} ({true_out}) - Pred: {label_class} ({label_out}) - Conf: {confidence:.2f}',
        fontsize=14
    )

    # input image
    ax0 = plt.subplot(gs[0])
    ax0.imshow(image)
    ax0.axis('off')
    ax0.set_title('Input Image')

    # conceptogram
    ax1 = plt.subplot(gs[1])
    ax1.imshow(concepto, aspect='auto', cmap='YlGnBu')
    ax1.set_xticks(np.arange(len(kwargs['target_layers'])))
    ax1.set_yticks([true_out, label_out], [f'{true_out} ({true_class})', f'{label_out} ({label_class})'])
    ax1.yaxis.tick_right()
    ax1.set_xlabel('Layers')
    ax1.set_title('Conceptogram')

    # network output 
    ax2 = plt.subplot(gs[2])
    ax2.imshow(out_net, cmap='YlGnBu', aspect='auto')
    ax2.set_xticks([])
    ax2.set_yticks([true_out])
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_xlabel('Net Out')
    ax2.set_title('Network Output')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save
    os.makedirs(save_path, exist_ok=True)
    filename = f'sample_conceptogram_{split}_sample{sample}_true{true_out}_pred{label_out}_conf{confidence:.2f}.png'
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file)
    plt.close()

    print(f"Conceptogram saved to: {save_file}")



def generate_sample_conceptograms(class_id, result, confidence, **kwargs):
    """
    Generates multiple detailed conceptograms (with network output) for a specific class, prediction result and minimum confidence.

    Args:
        sample(int): Class index to filter samples by. [0 to n_classes-1]
        result (bool or int): Desired prediction correctness (1 for correct, 0 for incorrect).
        confidence (float): Minimum confidence threshold to include sample.
        peepholes (Peepholes): Peepholes (already initialized).
        ds (Dataset): Dataset.
        corevecs (CoreVectors): Corevectors.
        save_path (str or Path): File path where the final figure will be saved.
        target_layers (list): List of model layers for which to extract peepholes.
        n_classes (int): Total number of output classes.
    """

    o_dnn, o_dnn_dfs = _extract_outputs(kwargs['corevecs'])
    split = 'test'
    save_path = kwargs['save_path']
    ds = kwargs['ds']   

    rand_samples = []
    k = 0

    if class_id >= len(ds._classes):
        raise ValueError(f"Class ID has to be a number between 0 and {len(ds._classes)-1}!")

    while len(rand_samples) < 5 and k < 500:
        if o_dnn_dfs[split]['result'][k] == result and \
           o_dnn_dfs[split]['true'][k] == class_id and \
           o_dnn_dfs[split]['max'][k] >= confidence:
            rand_samples.append(k)
        k += 1

    print(f'Samples found in split {split}: ', rand_samples)

    cgs_dict = _generate_conceptograms_dict(
        peepholes=kwargs['peepholes'],
        target_layers=kwargs['target_layers'],
        n_classes=kwargs['n_classes'],
    )


    # Generate conceptograms for each sample
    fig, axes = plt.subplots(1, 10, figsize=(30, 10))

    for i, sample_val in enumerate(rand_samples):
        true_out = o_dnn_dfs[split]['true'][sample_val]
        true_class = ds._classes[true_out]

        label_out = o_dnn_dfs[split]['label'][sample_val]
        label_class = ds._classes[label_out]

        confidence = o_dnn_dfs[split]['max'][sample_val]

        concepto = cgs_dict[split][sample_val].detach().cpu().numpy().T
        out_net = np.expand_dims(o_dnn[split][sample_val], axis=1)

        ax_concepto = axes[i * 2]
        ax_concepto.imshow(concepto, aspect='auto', cmap='YlGnBu')
        ax_concepto.set_xticks(np.arange(len(kwargs['target_layers'])))
        ax_concepto.set_yticks([true_out, label_out], [f'{true_out} ({true_class})', f'{label_out} ({label_class})'])
        ax_concepto.yaxis.tick_right()
        ax_concepto.set_xlabel('Layers')
        ax_concepto.set_title(f'Class ID: {class_id}\nConfidence: {confidence:.2f}')

        ax_out_net = axes[i * 2 + 1]
        ax_out_net.imshow(out_net, cmap='YlGnBu')
        ax_out_net.set_xticks([])
        ax_out_net.set_yticks([true_out])
        ax_out_net.yaxis.set_label_position("right")
        ax_out_net.yaxis.tick_right()
        ax_out_net.set_xlabel('Net Out')

    fig.suptitle(f'Split: {split} - Class: {class_id} ({ds._classes[class_id]})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save conceptograms
    os.makedirs(save_path, exist_ok=True)
    res_str = 'correct' if result else 'incorrect'
    filename = f'conceptograms_{split}_class{class_id}_{res_str}_conf{confidence:.2f}.png'
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file)
    plt.close()
    print(f"Saved conceptogram figure to: {save_path}")