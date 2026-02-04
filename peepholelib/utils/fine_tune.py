#general python stuff
from pathlib import Path
from matplotlib import pyplot as plt
from functools import partial
from math import ceil
from time import time
from tqdm import tqdm

# torch stuff
import torch
from torch.utils.data import DataLoader

def img_classification_acc(pred, target):
    
    pred_idx = torch.argmax(pred, dim=1)
    return (pred_idx == target).sum()

def get_trainable_params(model):
    """Get list of trainable parameters."""
    return [p for p in model.parameters() if p.requires_grad]

def fine_tune(**kwargs):
    model = kwargs['model']
    device = model.device
    path = Path(kwargs['path'])
    name = kwargs['name']

    # dataset 
    ds = kwargs['dataset']
    train_key = kwargs.get('train_key', 'train')
    val_key = kwargs.get('val_key', 'val')
    in_parser = kwargs.get('in_parser', lambda x:x)
    out_parser = kwargs.get('out_parser', lambda x:x)

    # training artifacts
    _l = kwargs.get('loss_fn', torch.nn.CrossEntropyLoss)
    loss_kwargs = kwargs.get('loss_kwargs', dict())
    _opt = kwargs.get('optimizer', torch.optim.SGD)
    optim_kwargs = kwargs.get('optim_kwargs', dict())
    acc_fn = kwargs.get('acc_fn', img_classification_acc)
    _sched = kwargs.get('scheduler', None)
    scheduler_kwargs = kwargs['scheduler_kwargs'] if 'scheduler_kwargs' in kwargs and 'scheduler' in kwargs else {} 
    
    # training progress
    lr = kwargs.get('lr', 5e-5)
    bs = kwargs.get('batch_size', 256)
    max_epochs = kwargs.get('max_epochs', 1000)
    iterations = kwargs.get('iterations', 'full')
    n_threads = kwargs.get('n_threads', 0)

    # early stopping
    early_stopping = kwargs.get('early_stopping', False)
    early_stopping_patience = kwargs.get('early_stopping_patience', 10)
    early_stopping_min_delta = kwargs.get('early_stopping_min_delta', 0.0)

    # Layer freezing configuration 
    freeze_all_but = kwargs.get('freeze_all_but', None)

    # saving
    save_every = kwargs.get('save_every', 100)
    verbose = kwargs.get('verbose', True)
    
    # create training artifacts
    loss_fn = _l(**loss_kwargs)

    # Apply initial layer freezing
    if freeze_all_but is not None:
        if verbose:
            print(f'Freezing all layers except: {freeze_all_but}')
        model.set_requires_grad(set_requires_grad = False, layer_names = None)
        model.set_requires_grad(set_requires_grad = True, layer_names = freeze_all_but)
    else:
        model.set_requires_grad(set_requires_grad = True, layer_names = None)
        if verbose:
            print(f'No layers to freeze. Training all layers')

    # Only pass trainable parameters to optimizer
    trainable_params = get_trainable_params(model._model)
    if verbose:
        total_params = sum(p.numel() for p in model._model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        print(f'Trainable parameters: {trainable_count:,} / {total_params:,} ({100*trainable_count/total_params:.2f}%)')
    
    optim = _opt(trainable_params, lr=lr, **optim_kwargs)
    scheduler = _sched(optimizer=optim, **scheduler_kwargs) if _sched is not None else None
    
    if iterations == 'full': 
        if verbose: print('using the whole dataset every iteration')
        iter_train = ceil(len(ds._dss[train_key])/bs)
        iter_val = ceil(len(ds._dss[val_key])/bs) 
    else:
        iter_train = iterations 
        iter_val = iterations 

    # dataloader for the dataset
    train_dl = DataLoader(
            dataset = ds._dss[train_key], 
            batch_size = bs, 
            shuffle = True, 
            collate_fn = lambda x:x, 
            num_workers = n_threads,
        )

    val_dl = DataLoader(
            dataset = ds._dss[val_key], 
            batch_size = bs, 
            shuffle = False, 
            collate_fn = lambda x:x, 
            num_workers = n_threads,
        ) 
    
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
        best_epoch = data['best_epoch']
        best_val_loss = data['best_val_loss']

        model.load_checkpoint(
                path = path,
                name = _f,
                vebose = verbose
                )
        
        initial_epoch = trained_for
        
        if data['freeze_all_but'] == freeze_all_but:
        
            optim.load_state_dict(data['optimizer']) 

            if scheduler is not None and 'scheduler' in data and data['scheduler'] is not None:
                current_state = scheduler.state_dict()
                saved_state = data['scheduler']

                same_state = current_state == saved_state
                if same_state: scheduler.load_state_dict(data['scheduler'])
          
    else:
        if verbose: print('No training ongoing, starting anew.')
        initial_epoch = 0
        best_val_loss = float('inf')
        best_epoch = 0

    path.mkdir(parents=True, exist_ok=True)
    best_model_path = path/'best_model'
    best_model_path.mkdir(parents=True, exist_ok=True)

    # training loop
    if verbose: print('training------')
    
    old_lr = lr

    for epoch in range(initial_epoch, max_epochs):
        
        t0 = time()

        # peform train iterations
        loss_acc = 0.0
        acc_acc = 0.0
        samples_acc = 0
        epochs_without_improvement = 0

        model._model.train()
        for it, _data in zip(range(iter_train), train_dl):
            data = in_parser(_data)
            images = data['image'].contiguous().to(device, non_blocking=True)
            labels = data['label'].contiguous().to(device, non_blocking=True)
            n_samples = len(images)
            samples_acc += n_samples 
            model_out = model(images)
            pred = out_parser(model_out)
            loss = loss_fn(pred, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_acc += loss*n_samples
            acc_acc += acc_fn(pred, labels)

        train_losses[epoch] = (loss_acc/samples_acc).detach().cpu()
        train_acc[epoch] = (acc_acc/samples_acc).detach().cpu()

        # validation
        model._model.eval()
        with torch.no_grad():

            loss_acc = 0.0
            acc_acc = 0.0
            samples_acc = 0

            for it, _data in zip(range(iter_val), val_dl):

                data = in_parser(_data)

                images = data['image'].contiguous().to(device, non_blocking=True)
                labels = data['label'].contiguous().to(device, non_blocking=True)
                n_samples = len(images)
                samples_acc += n_samples 

                model_out = model(images)
                pred = out_parser(model_out)
                loss = loss_fn(pred, labels)

                loss_acc += loss*n_samples
                acc_acc += acc_fn(pred, labels)

            val_losses[epoch] = (loss_acc/samples_acc).detach().cpu()
            val_acc[epoch] = (acc_acc/samples_acc).detach().cpu()

        # step the scheduler
        if scheduler is not None: 
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses[epoch])
                current_lr = optim.param_groups[0]['lr']

                if old_lr != current_lr:
                    old_lr = current_lr
                    if verbose:  print(f'New LR: {optim.param_groups[0]["lr"]:.6f}')
            else:
                scheduler.step()
        
        if val_losses[epoch] < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_losses[epoch]
            epochs_without_improvement = 0
            # Save best model
            _d = {
                  'epoch': epoch,
                  'train_losses': train_losses[:epoch+1],
                  'train_accuracy': train_acc[:epoch+1],
                  'val_losses': val_losses[:epoch+1],
                  'val_accuracy': val_acc[:epoch+1],
                  'state_dict': model._model.state_dict(),
                  'optimizer': optim.state_dict(),
                  'scheduler': scheduler.state_dict() if not scheduler == None else None,
                  'freezed_all_but': freeze_all_but,
                  'best_epoch': best_epoch,
                  'best_val_loss': best_val_loss
                  }
            torch.save(_d, best_model_path/'best_model_config.pt')
            best_epoch = epoch

            if verbose: print(f'  → New best validation loss: {best_val_loss:.6f}')

        else:
            epochs_without_improvement += 1

        if verbose: 
            print(f'epoch {epoch} - train loss: {train_losses[epoch]:.4f} - val loss: {val_losses[epoch]:.4f} - train acc: {train_acc[epoch]*100:.2f} - val acc: {val_acc[epoch]*100:.2f} - time: {time()-t0:.2f}')

        if early_stopping and epochs_without_improvement >= early_stopping_patience:
            if verbose:
                print(f'\nEarly stopping triggered after {early_stopping_patience} epochs without improvement')
            break
        
        # saving and plotting
        if (epoch+1)%save_every == 0:
            _d = {
                  'epoch': epoch,
                  'train_losses': train_losses[:epoch+1],
                  'train_accuracy': train_acc[:epoch+1],
                  'val_losses': val_losses[:epoch+1],
                  'val_accuracy': val_acc[:epoch+1],
                  'state_dict': model._model.state_dict(),
                  'optimizer': optim.state_dict(),
                  'scheduler': scheduler.state_dict() if not scheduler == None else None,
                  'freeze_all_but': freeze_all_but, 
                  'best_epoch': best_epoch,
                  'best_val_loss': best_val_loss
                  }
            torch.save(_d, file.as_posix()+'.'+str(epoch)+'.pt')

            fig, axs = plt.subplots(2, 1, figsize=(10,8))

            train_losses_np = train_losses[:epoch+1].detach().cpu().numpy()
            val_losses_np = val_losses[:epoch+1].detach().cpu().numpy()
            train_acc_np = train_acc[:epoch+1].detach().cpu().numpy()
            val_acc_np = val_acc[:epoch+1].detach().cpu().numpy()

            axs[0].plot(train_losses_np, label='loss_train')
            axs[0].plot(val_losses_np, label='loss_val')
            axs[0].set_ylabel('loss')
            axs[0].set_title('Loss')

            axs[1].plot(train_acc_np*100, label='train')
            axs[1].plot(val_acc_np*100, label='val')
            axs[1].set_ylabel('Acc')
            axs[1].set_xlabel('epoch')
            axs[1].set_title('Accuracy')

            # Highlight best model epoch with a star on each curve.

            axs[0].plot(
                [best_epoch], [train_losses_np[best_epoch]],
                marker='*', markersize=12, linestyle='None',
                color=axs[0].lines[0].get_color()
            )
            axs[0].plot(
                [best_epoch], [val_losses_np[best_epoch]],
                marker='*', markersize=12, linestyle='None',
                color=axs[0].lines[1].get_color(), label=f'best loss {val_losses_np[best_epoch]:.3f}'
            )
            axs[1].plot(
                [best_epoch], [train_acc_np[best_epoch]],
                marker='*', markersize=12, linestyle='None',
                color=axs[1].lines[0].get_color()
            )
            axs[1].plot(
                [best_epoch], [val_acc_np[best_epoch]],
                marker='*', markersize=12, linestyle='None',
                color=axs[1].lines[1].get_color(), label=f'best Acc {val_acc_np[best_epoch]:.3f}'
            )

            for ax in axs:
                ax.semilogy()
                ax.legend()

            fig.savefig(file.as_posix()+'.losses.png', dpi=300, bbox_inches='tight')