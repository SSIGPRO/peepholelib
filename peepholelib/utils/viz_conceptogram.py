import torch
import matplotlib.pyplot as plt

def get_conceptogram_class(cv, ph, idx, target_layers, portion, ticks, k_rows, list_classes, path=None):
    
    if len(target_layers) != len(ticks):
        raise ValueError('Number of target layers and ticks should be equal')
    
    output = cv._actds[portion]['output']
    SM = torch.nn.Softmax(dim=1)
    output = SM(output)[idx].unsqueeze(0)
    pred = cv._actds[portion][idx]['pred']
    lab = list_classes[int(cv._actds[portion][idx]['label'].detach().cpu().numpy())]

    conceptogram = torch.stack([ph._phs[portion][idx][layer]['peepholes'] for layer in target_layers])
    _, idx_topk = torch.topk(conceptogram.sum(dim=0), k_rows,sorted=False)
    
    classes_topk = [list_classes[i] for i in idx_topk.tolist()]
    tick_positions = idx_topk.cpu().tolist()

    # Create tick labels; for example, associating each tick with its class and index.
    tick_labels = [f'{cls} ({pos})' for cls, pos in zip(classes_topk, tick_positions)]

    fig, axs = plt.subplots(1,3, figsize=(10,13),  gridspec_kw={'width_ratios': [1, 2, 1]} )

    axs[0].imshow(cv._actds[portion][idx]['image'].permute(1,2,0).cpu())
    axs[0].set_title(f'Label  $\mathbf{{{lab}}}$')
    axs[0].axis('off')

    axs[1].imshow(conceptogram.detach().cpu().numpy().T, cmap='YlGnBu')
    axs[1].set_title('Conceptogram')
    axs[1].set_xticks(ticks=range(len(ticks)), labels=ticks, rotation=90, fontsize=8)
    axs[1].set_yticks(tick_positions, tick_labels)
    axs[1].yaxis.tick_right()
    axs[1].set_xlabel('VGG16 Layers')

    pred_class = list_classes[int(pred.detach().cpu().numpy())]
    tick_position = int(pred.detach().cpu().numpy())

    axs[2].imshow(output.detach().cpu().numpy().T, cmap='YlGnBu')
    axs[2].set_title(f'Prediction\n {pred_class} ({tick_position})')
    axs[2].set_yticks([tick_position])
    axs[2].set_yticklabels([f'Confidence: {torch.max(output)*100:.2f}%'])
    axs[2].yaxis.tick_right()
    axs[2].set_xticks([])
    fig.suptitle(f'Image Portion: {portion} index: {idx}')

    plt.tight_layout()
    plt.show()

    if path==None:
        path = f'conceptogram_{portion}_{idx}.png'

    fig.savefig(path, dpi=300, bbox_inches='tight')

    return

def get_conceptogram_superclass(cv, ph, idx, target_layers, portion, ticks, k_rows, list_classes, list_superclasses, path=None):
    
    if len(target_layers) != len(ticks):
        raise ValueError('Number of target layers and ticks should be equal')
    
    output = cv._actds[portion]['output']
    SM = torch.nn.Softmax(dim=1)
    output = SM(output)[idx].unsqueeze(0)
    pred = cv._actds[portion][idx]['pred']
    lab = list_classes[int(cv._actds[portion][idx]['label'].detach().cpu().numpy())]
    sup_class = list_superclasses[int(cv._actds[portion][idx]['superclass'].detach().cpu().numpy())]


    conceptogram = torch.stack([ph._phs[portion][idx][layer]['peepholes'] for layer in target_layers])
    _, idx_topk = torch.topk(conceptogram.sum(dim=0), k_rows,sorted=False)
    
    classes_topk = [list_superclasses[i] for i in idx_topk.tolist()]
    tick_positions = idx_topk.cpu().tolist()

    # Create tick labels; for example, associating each tick with its class and index.
    tick_labels = [f'{cls}' for cls in classes_topk]

    fig, axs = plt.subplots(1,3, figsize=(10,13),  gridspec_kw={'width_ratios': [1, 2, 1]} )

    axs[0].imshow(cv._actds[portion][idx]['image'].permute(1,2,0).cpu())
    axs[0].set_title(f'Label $\mathbf{{{lab}}}$\n Superclass $\mathbf{{{sup_class}}}$')
    axs[0].axis('off')

    axs[1].imshow(conceptogram.detach().cpu().numpy().T, cmap='YlGnBu')
    axs[1].set_title('Conceptogram')
    axs[1].set_xticks(ticks=range(len(ticks)), labels=ticks, rotation=90, fontsize=12, weight="bold")
    axs[1].set_yticks(tick_positions, tick_labels, fontsize=8)
    axs[1].yaxis.tick_right()
    axs[1].set_xlabel('VGG16 Layers', fontsize=12)

    pred_class = list_classes[int(pred.detach().cpu().numpy())]
    tick_position = int(pred.detach().cpu().numpy())

    axs[2].imshow(output.detach().cpu().numpy().T, cmap='YlGnBu')
    axs[2].set_title(f'Prediction\n {pred_class} ({tick_position})')
    axs[2].set_yticks([tick_position])
    axs[2].set_yticklabels([f'Confidence: {torch.max(output)*100:.2f}%'])
    axs[2].yaxis.tick_right()
    axs[2].set_xticks([])
    fig.suptitle(f'Label $\mathbf{{{lab}}}$\n Superclass $\mathbf{{{sup_class}}}$')
    fig.suptitle(f'Image Portion: {portion} index: {idx}')

    plt.tight_layout()
    plt.show()

    if path==None:
        path = f'conceptogram_{portion}_{idx}.png'

    fig.savefig(path, dpi=300, bbox_inches='tight')

    return
