import torch

def img_classification_full(**kwargs):
    model = kwargs['model']
    data = kwargs['data']
    
    device = model.device

    d_in = data['image'].to(device)

    out = model(d_in)
    pred = torch.argmax(out,axis=1)
    res = pred == data['label'].to(device) 

    ret = {
            'output': out,
            'pred': pred,
            'result': res 
            }

    return ret
