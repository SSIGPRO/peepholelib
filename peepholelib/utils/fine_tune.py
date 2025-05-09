#general python stuff
from pathlib import Path
from matplotlib import pyplot as plt
from functools import partial

# torch stuff
import torch
from torch.utils.data import DataLoader

# our stuff
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.datasets.dataset_base import DatasetBase 
from peepholelib.coreVectors.parsers import from_dataset
from peepholelib.coreVectors.prediction_fns import multilabel_classification

def fine_tune(**kwargs):
    model = kwargs['model']
    device = model.device

    path = Path(kwargs['path'])
    name = kwargs['name']
    
    # dataset 
    ds = kwargs['dataset']
    train_key = kwargs['train_key'] if 'train_key' in kwargs else 'train' 
    val_key = kwargs['val_key'] if 'val_key' in kwargs else 'val' 
    ds_parser = kwargs['ds_parser'] if 'ds_parser' in kwargs else from_dataset 
    
    # training
    lr = kwargs['lr']
    _l = kwargs['loss_fn'] if 'loss_fn' in kwargs else torch.nn.CrossEntropyLoss  
    loss_fn = _l()
    iterations = kwargs['iterations'] if 'iterations' in kwargs else 1 
    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 'full'
    max_epochs = kwargs['max_epochs'] if 'max_epochs' in kwargs else 1e3
    save_every = kwargs['save_every'] if 'save_every' in kwargs else 100
    _opt = kwargs['optimizer'] if 'optimizer' in kwargs else torch.optim.SGD
    optim_kwargs = kwargs['optim_kwargs'] if 'optim_kwargs' in kwargs else dict()
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    
    # TODO add lr scheduling

    assert(isinstance(model, ModelWrap))
    assert(isinstance(ds, DatasetBase))
    assert(isinstance(bs, int) or bs == 'full')
    
    # construct optimizer
    optim = _opt(model._model.parameters(), lr=lr, **optim_kwargs)

    if bs == 'full': 
        if verbose: print('using the whole dataset every iteration')
        bs = len(ds._dss[train_key]) 

    # dataloader for the dataset
    train_dl = DataLoader(dataset=ds._dss[train_key], batch_size=bs, collate_fn=partial(ds_parser), shuffle=True)
    val_dl = DataLoader(dataset=ds._dss[val_key], batch_size=len(ds._dss[val_key]), collate_fn=partial(ds_parser), shuffle=True) 
    
    # to save losses
    train_losses = torch.zeros(max_epochs)
    val_losses = torch.zeros(max_epochs)
    
    #TODO: load model and training data

    # start training from this epoch
    initial_epoch = 0

    # training loop
    if verbose: print('training------')
    for epoch in range(initial_epoch, max_epochs):
        # peform train iterations
        loss_acc = 0.0
        for it, data in zip(range(iterations), train_dl):
            pred = model(data['image'].to(device))
            print('pred: ', pred.dtype)
            labels = data['label'].to(device)
            print('labels: ', labels.dtype)
            loss = loss_fn(pred, labels)
            loss_acc += loss
            optim.zero_grad()
            loss.backward()
            optim.step()
        train_losses[epoch] = loss_acc/max_epochs
        
        # validation
        with torch.no_grad():
            loss_acc = 0.0
            for data in val_dl:
                pred = model(data['image'].to(device))
                labels = data['label'].to(device)
                loss = loss_fn(pred, labels)
                loss_acc += loss
        val_losses[epoch] = loss_acc/max_epochs
        
        if verbose: print(f'epoch `{epoch} - train loss: {train_losses[epoch]} - val loss: {val_losses[epoch]}')
        # TODO: save move on epoch%save_every == 0

    # saving and plotting
    path.mkdir(parents=True, exist_ok=True)
    torch.save([train_losses, val_losses], path/(name+'.losses.pt'))
    plt.figure()
    plt.plot(train_losses.detach().cpu().numpy(), label='train')
    plt.plot(val_losses.detach().cpu().numpy(), label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(path/(name+'.losses.png'), dpi=300, bbox_inches='tight')
