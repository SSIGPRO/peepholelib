#general python stuff
from pathlib import Path as Path
import abc 
from matplotlib import pyplot as plt
from functools import partial
from math import ceil
from time import time
from math import isinf

# torch stuff
import torch
from torch.utils.data import DataLoader

from peepholelib.training.trainingLoops import DefaultTrainLoop
from peepholelib.training.validationLoops import DefaultValidationLoop
from peepholelib.training.testingLoops import DefaultTestLoop
from peepholelib.training.savingLoops import DefaultSavingLoop

def img_classification_acc(pred, target):
    
    pred_idx = torch.argmax(pred, dim=1)
    return (pred_idx == target).sum()

class Trainer(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        """
        Base trainer that owns the full training state and orchestrates the
        training/validation/testing workflow via pluggable loop strategies.

        It builds dataloaders, tracks losses/accuracies, handles checkpoint
        resume, early stopping, and scheduler stepping. The actual per-epoch
        logic is delegated to `train_loop`, `validation_loop`, and
        `saving_loop`, which can be swapped to customize behavior without
        rewriting the whole trainer.

        Args:
        - model (modelWrap): model wrapper to train/evaluate
        - path: directory where checkpoints and plots are stored
        - name: base filename for checkpoints
        - verbose: log progress to stdout

        - dataset: dataset container with `__dataset__` dict
        - train_key: key for train split in `__dataset__`
        - val_key: key for validation split in `__dataset__`
        - in_parser: function to map raw batch -> dict with tensors
        - out_parser: function to map model output -> prediction tensor
        - batch_size: batch size for train/val loaders
        - dataloader_kwargs: extra kwargs for `torch.utils.data.DataLoader`
        - iterations: "full" or int number of iterations per epoch
        - loss_fn/loss_kwargs: loss class and kwargs
        - acc_fn: accuracy function
        - optimizer: torch optimizer
        - scheduler: optional LR scheduler
        - max_epochs: maximum number of epochs to run
        - early_stopping: enable early stopping when scheduler supports it
        - train_loop/validation_loop/saving_loop: pluggable loop objects
        """

        # Preliminaries
        self.model = kwargs['model']
        self.device = self.model.device
        self.path = Path(kwargs['path'])
        self.name = kwargs['name']
        self.verbose = kwargs.get('verbose', True)

        self.file = self.path/self.name

        # Dataset
        self.ds = kwargs['dataset']
        self.train_key = kwargs.get('train_key', 'train')
        self.val_key = kwargs.get('val_key', 'val')
        self.test_key = kwargs.get('test_key', 'test')
        self.in_parser = kwargs.get('in_parser', lambda x:x)
        self.out_parser = kwargs.get('out_parser', lambda x:x)

        # Dataloader
        bs = kwargs['batch_size']
        dl_kwargs = kwargs['dataloader_kwargs']
        iterations = kwargs['iterations']

        # Loss function
        _l = kwargs.get('loss_fn', torch.nn.CrossEntropyLoss)
        loss_kwargs = kwargs.get('loss_kwargs', dict())
        self.acc_fn = kwargs.get('acc_fn', img_classification_acc)

        self.loss_fn = _l(**loss_kwargs)

        # Training artifacts
        self.max_epochs = kwargs.get('max_epochs', 1000)
        self.early_stopping = kwargs.get('early_stopping', True)
        self.optim = kwargs['optimizer']
        self.scheduler = kwargs.get('scheduler', None)
        self.train_loop = kwargs.get("train_loop", DefaultTrainLoop)
        self.validation_loop = kwargs.get("validation_loop", DefaultValidationLoop)
        self.test_loop = kwargs.get("test_loop", DefaultTestLoop)
        self.saving_loop = kwargs.get("saving_loop", DefaultSavingLoop)
        self.save_every = kwargs.get("save_every", 10)
        self.phase_num = kwargs.get("phase_num", None)
        self.early_stopping_patience = kwargs.get("early_stopping_patience", float('inf'))
        self._plot_archived = False
        self.no_training = False
        self.bad_epochs = 0

        self.train_dl = DataLoader(
                                dataset=self.ds.__dataset__[self.train_key],
                                batch_size=bs,
                                shuffle=True,
                                **dl_kwargs,
                            )

        self.val_dl = DataLoader(
                                dataset=self.ds.__dataset__[self.val_key],
                                batch_size=bs,
                                shuffle=False,
                                **dl_kwargs,
                            ) 
        
        self.test_dl = DataLoader(
                                dataset=self.ds.__dataset__[self.test_key],
                                batch_size=bs,
                                shuffle=False,
                                **dl_kwargs,
                            ) 
        
        if iterations == 'full': 
            if self.verbose: print('using the whole dataset every iteration')
            self.iter_train = ceil(len(self.ds.__dataset__[self.train_key])/bs)
            self.iter_val = ceil(len(self.ds.__dataset__[self.val_key])/bs) 
        else:
            self.iter_train = iterations 
            self.iter_val = iterations 

        if self.early_stopping:
            if self.scheduler is None:
                raise ValueError('early_stopping=True requires a scheduler with num_bad_epochs and patience.')
            if not hasattr(self.scheduler, 'num_bad_epochs') or not hasattr(self.scheduler, 'patience'):
                if isinf(self.early_stopping_patience):
                    raise ValueError("early_stopping_patience cannot be infinite.")

        # Pre-allocate training history buffers
        self.train_losses = torch.zeros(self.max_epochs, requires_grad=False)
        self.val_losses = torch.zeros(self.max_epochs, requires_grad=False)
        self.train_acc = torch.zeros(self.max_epochs, requires_grad=False)
        self.val_acc = torch.zeros(self.max_epochs, requires_grad=False)

        # Check model existance
        ckps = sorted(list(self.path.glob('*.pt')))
        if self.path.exists() and len(ckps) > 0:
            ckps_n = [int(ckp.as_posix().replace(self.file.as_posix()+'.','').replace('/', '').replace('.pt','')) for ckp in ckps]
            trained_for = max(ckps_n)+1
            
            if trained_for >= self.max_epochs:
                print(f'Already trained for {trained_for} epochs, not doing anything.')
                self.no_training = True
                return
            else:
                if self.verbose: print(f'Found latest checkpoint for epoch {trained_for}. Resume training')
            
            _f = (self.path / 'best_model' / 'best_model_config.pt').as_posix()
            if self.verbose: print(f'Loading best model config from {_f}')
            data = torch.load(_f) 
            
            # to save accuracies and losses
            saved_len = len(data['train_losses'])
            self.train_losses[:saved_len] = data['train_losses']
            self.val_losses[:saved_len] = data['val_losses'] 
            self.train_acc[:saved_len] = data['train_accuracy']
            self.val_acc[:saved_len] = data['val_accuracy'] 
            self.best_epoch = data['best_epoch']
            self.best_val_loss = data['best_val_loss']

            self.model.load_checkpoint(
                    path = self.path,
                    name = _f,
                    vebose =self. verbose
                    )
            
            # resume from the checkpoint we loaded
            self.initial_epoch = int(data.get('best_epoch', trained_for-1)) + 1
            
            
            current_state = self.optim.state_dict()
            saved_state = data['optimizer']
            if current_state == saved_state:
                self.optim.load_state_dict(data['optimizer'])

            if self.scheduler is not None and 'scheduler' in data and data['scheduler'] is not None:
                current_state = self.scheduler.state_dict()
                saved_state = data['scheduler']

                if current_state == saved_state: self.scheduler.load_state_dict(data['scheduler'])
            
        else:
            if self.verbose: print('No training ongoing, starting anew.')
            self.initial_epoch = 0
            self.best_val_loss = float('inf')
            self.best_epoch = 0

        self.path.mkdir(parents=True, exist_ok=True)
        best_model_path = self.path/'best_model'
        best_model_path.mkdir(parents=True, exist_ok=True)

        if self.phase_num is None:
            # Auto-increment phase based on existing archived plots.
            try:
                phase_glob = f"{self.name}.losses.prev.phase_*.png"
                existing = list(self.path.glob(phase_glob))
                if existing:
                    nums = []
                    for p in existing:
                        stem = p.stem  # e.g. name.losses.prev.phase_3
                        if "phase_" in stem:
                            try:
                                nums.append(int(stem.split("phase_")[-1]))
                            except ValueError:
                                pass
                    self.phase_num = (max(nums) + 1) if nums else 1
                else:
                    self.phase_num = 1
            except Exception:
                self.phase_num = 1
    
    def fit(self):
        if self.no_training: return

        if self.verbose: print('----- Training Model ----- ')

        for epoch in range(self.initial_epoch, self.max_epochs):

            t0 = time()
            self.train_loop(self, epoch)
            stop = self.validation_loop(self, epoch, t0=t0)
            if stop:
                break
            if (epoch + 1) % self.save_every == 0:
                self.saving_loop(self, epoch)
                
    def test(self):
        if self.verbose: print('----- Testing Model ----- ')


        best_model_file = self.path / 'best_model' / 'best_model_config.pt'
        
        if self.verbose: print(f'Loading best model config from {best_model_file.as_posix()}')
        self.model.load_checkpoint(
                            path=best_model_file.parent,
                            name=best_model_file.name,
                            verbose=self.verbose,
                        )

        return self.test_loop(self)

    
