import torch

def img_classification_full(**kwargs):
    data = kwargs['data']
    model = kwargs['model']
    
    device = model.device

    d_in = data['image'].to(device)

    with torch.no_grad():
        out = model(d_in)

    pred = out.argmax(axis=1)
    res = pred == data['label'].to(device) 

    ret = {
            'output': out,
            'pred': pred,
            'result': res 
            }

    return ret


def img_classification_corruptions(**kwargs):
    data = kwargs['data']
    corrupt = kwargs['corruption']         
    model = kwargs['model']
    spatial_transform = kwargs['spatial_transform']  
    device = model.device

    imgs_native = data['image']
    labels = data['label'].to(device)

    imgs_corr, corruption_ids = corrupt(images=imgs_native, labels=labels)

    imgs_ori  = torch.stack([spatial_transform(img) for img in imgs_native]).to(device)
    imgs_corr = torch.stack([spatial_transform(img) for img in imgs_corr]).to(device)

    with torch.no_grad():
        out_ori  = model(imgs_ori)
        out_corr = model(imgs_corr)

    pred_ori  = out_ori.argmax(1)
    pred_corr = out_corr.argmax(1)

    return {
        'image': imgs_corr,
        'output': out_corr,
        'pred': pred_corr,
        'result': pred_corr == labels,
        'corruption': corruption_ids.to(device),
        'corruption_success': torch.logical_and(pred_ori == labels, pred_corr != pred_ori),
    }
def img_classification_atks(**kwargs):
    data = kwargs['data']
    atk = kwargs['attack']
    label_key = kwargs.get('label_key', 'label')
    verbose = kwargs.get('verbose', False)
    
    model = atk.model
    device = model.device

    with torch.enable_grad():
        imgs_ori = data['image'].to(device)
        labels = data[label_key].to(device)

        imgs_atk = atk(
                images = imgs_ori,
                labels = labels 
                )

    with torch.no_grad():
        out_ori = model(imgs_ori) 
        out_atk = model(imgs_atk)

    pred_ori = out_ori.argmax(axis=1)
    pred_atk = out_atk.argmax(axis=1)

    ret = {
            'image': imgs_atk,
            'output': out_atk,
            'pred': pred_atk,
            'result': pred_atk == labels, 
            'attack_success': torch.logical_and(pred_ori == labels, pred_atk != pred_ori)
            }
    if verbose:
        n_success = ret['attack_success'].sum().item()
        n_correct = (pred_ori == labels).sum().item()
        if n_correct > 0:
            print(f'Attacks successful: {int(n_success)}/{int(n_correct)} ({100 * n_success / n_correct:.1f}%)')

    return ret
