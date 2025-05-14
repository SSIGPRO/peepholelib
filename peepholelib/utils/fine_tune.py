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
    n_threads = kwargs['n_threads'] if 'n_threads' in kwargs else 1 

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
        bs_train = len(ds._dss[train_key]) 
        bs_val = len(ds._dss[val_key]) 
    else:
        bs_train = bs
        bs_val = bs 

    # dataloader for the dataset
    train_dl = DataLoader(dataset=ds._dss[train_key], batch_size=bs_train, collate_fn=partial(ds_parser), shuffle=True, num_workers=n_threads)
    val_dl = DataLoader(dataset=ds._dss[val_key], batch_size=bs_val, collate_fn=partial(ds_parser), shuffle=True, num_workers=n_threads) 
    
    # to save losses
    file = path/name

    train_losses = torch.zeros(max_epochs, requires_grad=False)
    val_losses = torch.zeros(max_epochs, requires_grad=False)
    
    # load model and training data
    if models_path.exists():
        ckps = sorted(list(models_path.glob('*.pt')))
        _mp_posix = models_path.as_posix()
        ckps_n = [int(ckp.as_posix().replace(_mp_posix,'').replace('/', '').replace('.pt','')) for ckp in ckps]
        trained_for = max(ckps_n)+1
        
        if trained_for >= max_epochs:
            print(f'Already trained for {trained_for} epochs, not doing anything.')
            return
        else:
            if verbose: print(f'Found latest checkpoint for epoch {trained_for}. Resume training')
        
        data = torch.load(file+'.'+str() 
        _tl, _vl = torch.load(losses_file)
        train_losses[:trained_for] = _tl[:trained_for]
        val_losses[:trained_for] = _tl[:trained_for]

        _m_name = 'model.'str(trained_for-1)+'.pt'
        _o_name = 'opt.'str(trained_for-1)+'.pt'
        if verbose: print(f'Loading {(models_path/_m_name).as_posix()}')
        model.load_checkpoint(
                path = models_path,
                name = _m_name,
                vebose = verbose
                )
        initial_epoch = trained_for 
    else:
        if verbose: print('No training ongoing, starting anew.')
        initial_epoch = 0

    models_path.mkdir(parents=True, exist_ok=True)

    # training loop
    if verbose: print('training------')
    for epoch in range(initial_epoch, max_epochs):
        # peform train iterations
        loss_acc = 0.0
        for it, data in zip(range(iterations), train_dl):
            pred = model(data['image'].to(device))
            labels = data['label'].to(device)
            loss = loss_fn(pred, labels)
            loss_acc += loss
            optim.zero_grad()
            loss.backward()
            optim.step()
        train_losses[epoch] = (loss_acc/max_epochs).detach().cpu()
        
        # validation
        with torch.no_grad():
            loss_acc = 0.0
            for it, data in zip(range(iterations), val_dl):
                pred = model(data['image'].to(device))
                labels = data['label'].to(device)
                loss = loss_fn(pred, labels)
                loss_acc += loss
            val_losses[epoch] = (loss_acc/max_epochs).detach().cpu()
        
        if verbose: print(f'epoch `{epoch} - train loss: {train_losses[epoch]} - val loss: {val_losses[epoch]}')
        
        # saving and plotting
        if (epoch+1)%save_every == 0:
            torch.save(model._model.state_dict(), models_path/f'{epoch}.pt')

            plt.figure()
            plt.plot(train_losses.detach().cpu().numpy(), label=train_key)
            plt.plot(val_losses.detach().cpu().numpy(), label=val_key)
            plt.semilogy()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        # save losses every epoch
        torch.save([train_losses, val_losses], losses_file)
