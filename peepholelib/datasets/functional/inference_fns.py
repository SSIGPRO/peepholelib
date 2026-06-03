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

def img_classification_atks(**kwargs):
    data = kwargs['data']
    atk = kwargs['attack']
    label_key = kwargs.get('label_key', 'label')
    
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

    return ret
