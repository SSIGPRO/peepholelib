#general python stuff
from pathlib import Path
from matplotlib import pyplot as plt
from functools import partial
from math import ceil
from time import time

# torch stuff
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel

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

    n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 1 
    devices = kwargs['devices'] if 'devices' in kwargs else None 
    if not devices == None: 
        model._model = DataParallel(model._model, device_ids=[i for i in devices]) 

    # dataset 
    ds = kwargs['dataset']
    train_key = kwargs['train_key'] if 'train_key' in kwargs else 'train' 
    val_key = kwargs['val_key'] if 'val_key' in kwargs else 'val' 
    train_dl_kwargs = kwargs['train_dl_kwargs'] if 'train_dl_kwargs' in kwargs else dict(batch_size=256, collate_fn= partial(from_dataset), shuffle=True, num_workers=n_threads)
    val_dl_kwargs = kwargs['val_dl_kwargs'] if 'val_dl_kwargs' in kwargs else train_dl_kwargs

    # training artifacts
    _l = kwargs['loss_fn'] if 'loss_fn' in kwargs else torch.nn.CrossEntropyLoss
    loss_kwargs = kwargs['loss_kwargs'] if 'loss_kwargs' in kwargs else dict()
    
    _opt = kwargs['optimizer'] if 'optimizer' in kwargs else torch.optim.SGD
    optim_kwargs = kwargs['optim_kwargs'] if 'optim_kwargs' in kwargs else dict()
    pred_fn = kwargs['pred_fn'] if 'pred_fn' in kwargs else partial(torch.argmax, axis=1)  
    _sched = kwargs['scheduler'] if 'scheduler' in kwargs else None 
    scheduler_kwargs = kwargs['scheduler_kwargs'] if 'scheduler_kwargs' in kwargs and 'scheduler' in kwargs else {} 
    
    # training progress
    lr = kwargs['lr']

    patience = kwargs['patience'] if 'patience' in kwargs else 10
    iterations = kwargs['iterations'] if 'iterations' in kwargs else 'full' 
    bs = kwargs['batch_size'] if 'batch_size' in kwargs else 256 
    max_epochs = kwargs['max_epochs'] if 'max_epochs' in kwargs else 1e3
    
    # create training artifacts
    loss_fn = _l(**loss_kwargs)
    optim = _opt(model._model.parameters(), lr=lr, **optim_kwargs)
    scheduler = _sched(optimizer=optim, **scheduler_kwargs) if not _sched == None else None

    # saving
    save_every = kwargs['save_every'] if 'save_every' in kwargs else 100
    verbose = kwargs['verbose'] if 'verbose' in kwargs else False
    
    # TODO add lr scheduling

    if iterations == 'full': 
        if verbose: print('using the whole dataset every iteration')
        iter_train = ceil(len(ds._dss[train_key])/bs)
        iter_val = ceil(len(ds._dss[val_key])/bs) 
    else:
        iter_train = iterations 
        iter_val = iterations 

    # dataloader for the dataset
    train_dl = DataLoader(dataset=ds._dss[train_key], **train_dl_kwargs)
    val_dl = DataLoader(dataset=ds._dss[val_key], **val_dl_kwargs) 
    
    # to save losses
    file = path/name

    train_losses = torch.zeros(max_epochs, requires_grad=False)
    val_losses = torch.zeros(max_epochs, requires_grad=False)
    train_acc = torch.zeros(max_epochs, requires_grad=False)
    val_acc = torch.zeros(max_epochs, requires_grad=False)
    
    # load model and training data
    ckps = sorted(list(path.glob('*.pt')))
    if path.exists() and len(ckps) > 0:
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
        if not scheduler == None:
            scheduler.load_state_dict(data['scheduler'])

        initial_epoch = trained_for 
    else:
        if verbose: print('No training ongoing, starting anew.')
        initial_epoch = 0

    path.mkdir(parents=True, exist_ok=True)
    best_model_path = path/'best_model'
    best_model_path.mkdir(parents=True, exist_ok=True)

    # training loop
    if verbose: print('training------')
    best_val_loss = float('inf')
    patience_counter = 0
    old_lr = lr

    for epoch in range(initial_epoch, max_epochs):
        t0 = time()
        # peform train iterations
        loss_acc = 0.0
        acc_acc = 0.0
        samples_acc = 0
        for it, data in zip(range(iter_train), train_dl):
            n_samples = len(data['image'])
            samples_acc += n_samples 
            pred = model(data['image'].to(device))
            labels = data['label'].to(device)
            loss = loss_fn(pred, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_acc += loss*n_samples
            acc_acc += torch.count_nonzero(pred_fn(pred)==labels)
        train_losses[epoch] = (loss_acc/samples_acc).detach().cpu()
        train_acc[epoch] = (acc_acc/samples_acc).detach().cpu()

        # validation
        with torch.no_grad():
            loss_acc = 0.0
            acc_acc = 0.0
            samples_acc = 0
            for it, data in zip(range(iter_val), val_dl):
                n_samples = len(data['image'])
                samples_acc += n_samples 
                pred = model(data['image'].to(device))
                labels = data['label'].to(device)
                loss = loss_fn(pred, labels)
                loss_acc += loss*n_samples
                acc_acc += torch.count_nonzero(pred_fn(pred)==labels)
            val_losses[epoch] = (loss_acc/samples_acc).detach().cpu()
            val_acc[epoch] = (acc_acc/samples_acc).detach().cpu()
        # step the scheduler
        if not scheduler == None: 
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
                if old_lr != optim.param_groups[0]['lr']:
                    old_lr = optim.param_groups[0]['lr']
                    if verbose: 
                        print(f'Current LR: {optim.param_groups[0]["lr"]:.6f}')
            else:
                scheduler.step()
        
        if loss < best_val_loss:
            best_val_loss = loss
            patience_counter = 0
            # Save best model
            _d = {
                  'train_losses': train_losses[:epoch+1],
                  'train_accuracy': train_acc[:epoch+1],
                  'val_losses': val_losses[:epoch+1],
                  'val_accuracy': val_acc[:epoch+1],
                  'state_dict': model._model.state_dict(),
                  'optimizer': optim.state_dict(),
                  'scheduler': scheduler.state_dict() if not scheduler == None else None
                  }
            torch.save(_d, best_model_path/'best_model_config.pt')
            torch.save(model._model.state_dict(), best_model_path/'best_model.pth')

        if verbose: print(f'epoch `{epoch} - train loss: {train_losses[epoch]} - val loss: {val_losses[epoch]} - train acc: {train_acc[epoch]} - val acc: {val_acc[epoch]} - time: {time()-t0}')
        
        # saving and plotting
        if (epoch+1)%save_every == 0:
            _d = {
                  'train_losses': train_losses[:epoch+1],
                  'train_accuracy': train_acc[:epoch+1],
                  'val_losses': val_losses[:epoch+1],
                  'val_accuracy': val_acc[:epoch+1],
                  'state_dict': model._model.state_dict(),
                  'optimizer': optim.state_dict(),
                  'scheduler': scheduler.state_dict() if not scheduler == None else None
                  }
            torch.save(_d, file.as_posix()+'.'+str(epoch)+'.pt')

            plt.figure()
            plt.plot(train_losses[:epoch-1].detach().cpu().numpy(), label='loss_'+train_key)
            plt.plot(val_losses[:epoch-1].detach().cpu().numpy(), label='loss_'+val_key)
            plt.plot(train_acc[:epoch-1].detach().cpu().numpy(), label='acc_'+train_key)
            plt.plot(val_acc[:epoch-1].detach().cpu().numpy(), label='acc_'+val_key)
            plt.semilogy()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(file.as_posix()+'.losses.png', dpi=300, bbox_inches='tight')
