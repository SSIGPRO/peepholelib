#general python stuff
from pathlib import Path
from matplotlib import pyplot as plt
from functools import partial
from math import ceil

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
    n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 1 

    # training artifacts
    _l = kwargs['loss_fn'] if 'loss_fn' in kwargs else torch.nn.CrossEntropyLoss  
    loss_kwargs = kwargs['loss_kwargs'] if 'loss_kwargs' in kwargs else {'reduction': 'mean'}
    _opt = kwargs['optimizer'] if 'optimizer' in kwargs else torch.optim.SGD
    optim_kwargs = kwargs['optim_kwargs'] if 'optim_kwargs' in kwargs else dict()
    pred_fn = kwargs['pred_fn'] if 'pred_fn' in kwargs else partial(torch.argmax, axis=1)  

    # training progress
    lr = kwargs['lr']
    iterations = kwargs['iterations'] if 'iterations' in kwargs else 'full' 
    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 256 
    max_epochs = kwargs['max_epochs'] if 'max_epochs' in kwargs else 1e3
    
    # create training artifacts
    loss_fn = _l()
    optim = _opt(model._model.parameters(), lr=lr, **optim_kwargs)

    # saving
    save_every = kwargs['save_every'] if 'save_every' in kwargs else 100
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    
    # TODO add lr scheduling

    assert(isinstance(model, ModelWrap))
    assert(isinstance(ds, DatasetBase))
    assert(isinstance(iterations, int) or iterations == 'full')
    

    if iterations == 'full': 
        if verbose: print('using the whole dataset every iteration')
        iter_train = ceil(len(ds._dss[train_key])/bs)
        iter_val = ceil(len(ds._dss[val_key])/bs) 
    else:
        iter_train = iterations 
        iter_val = iterations 

    # dataloader for the dataset
    train_dl = DataLoader(dataset=ds._dss[train_key], batch_size=bs, collate_fn=partial(ds_parser), shuffle=True, num_workers=n_threads)
    val_dl = DataLoader(dataset=ds._dss[val_key], batch_size=bs, collate_fn=partial(ds_parser), shuffle=True, num_workers=n_threads) 
    
    # to save losses
    file = path/name

    train_losses = torch.zeros(max_epochs, requires_grad=False)
    val_losses = torch.zeros(max_epochs, requires_grad=False)
    train_acc = torch.zeros(max_epochs, requires_grad=False)
    val_acc = torch.zeros(max_epochs, requires_grad=False)
    
    # load model and training data
    if path.exists():
        ckps = sorted(list(path.glob('*.pt')))
        ckps_n = [int(ckp.as_posix().replace(file.as_posix()+'.','').replace('/', '').replace('.pt','')) for ckp in ckps]
        trained_for = max(ckps_n)+1
        
        if trained_for >= max_epochs:
            print(f'Already trained for {trained_for} epochs, not doing anything.')
            return
        else:
            if verbose: print(f'Found latest checkpoint for epoch {trained_for}. Resume training')
        
        _f = file.as_posix()+'.'+str(trained_for-1)+'.pt'
        if verbose: print(f'Loading {_f}')
        data = torch.load(_f) 
        
        # to save accuracies and losses
        train_losses[:trained_for] = data['train_losses']
        val_losses[:trained_for] = data['val_losses'] 
        train_acc[:trained_for] = data['train_accuracy']
        val_acc[:trained_for] = data['val_accuracy'] 

        model.load_checkpoint(
                path = path,
                name = _f,
                vebose = verbose
                )
        optim.load_state_dict(data['optimizer']) 

        initial_epoch = trained_for 
    else:
        if verbose: print('No training ongoing, starting anew.')
        initial_epoch = 0

    path.mkdir(parents=True, exist_ok=True)

    # training loop
    if verbose: print('training------')
    for epoch in range(initial_epoch, max_epochs):
        # peform train iterations
        loss_acc = 0.0
        acc_acc = 0.0
        samples_acc = 0
        for it, data in zip(range(iter_train), train_dl):
            samples_acc += len(data)
            pred = model(data['image'].to(device))
            labels = data['label'].to(device)
            loss = loss_fn(pred, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_acc += loss*len(data)
            acc_acc += torch.count_nonzero(pred_fn(pred)==labels)
        train_losses[epoch] = (loss_acc/samples_acc).detach().cpu()
        train_acc[epoch] = (acc_acc/samples_acc).detach().cpu()

        # validation
        with torch.no_grad():
            loss_acc = 0.0
            acc_acc = 0.0
            samples_acc = 0
            for it, data in zip(range(iter_val), val_dl):
                samples_acc += len(data)
                pred = model(data['image'].to(device))
                labels = data['label'].to(device)
                loss = loss_fn(pred, labels)
                loss_acc += loss*len(data)
                acc_acc += torch.count_nonzero(pred_fn(pred)==labels)
            val_losses[epoch] = (loss_acc/iter_val).detach().cpu()
            val_acc[epoch] = (acc_acc/samples_acc).detach().cpu()

        if verbose: print(f'epoch `{epoch} - train loss: {train_losses[epoch]} - val loss: {val_losses[epoch]}')
        
        # saving and plotting
        if (epoch+1)%save_every == 0:
            _d = {
                  'train_losses': train_losses[:epoch+1],
                  'train_accuracy': train_acc[:epoch+1],
                  'val_losses': val_losses[:epoch+1],
                  'val_accuracy': val_acc[:epoch+1],
                  'state_dict': model._model.state_dict(),
                  'optimizer': optim.state_dict()
                  }
            torch.save(_d, file.as_posix()+'.'+str(epoch)+'.pt')

            plt.figure()
            plt.plot(train_losses.detach().cpu().numpy(), label='loss_'+train_key)
            plt.plot(val_losses.detach().cpu().numpy(), label='loss_'+val_key)
            plt.plot(train_acc.detach().cpu().numpy(), label='acc_'+train_key)
            plt.plot(val_acc.detach().cpu().numpy(), label='acc_'+val_key)
            plt.semilogy()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(file.as_posix()+'.losses.png', dpi=300, bbox_inches='tight')
