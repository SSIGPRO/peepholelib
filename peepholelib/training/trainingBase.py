#general python stuff
from pathlib import Path as Path
from math import ceil
from time import time
from functools import partial
import re

# torch stuff
import torch
from torch.utils.data import DataLoader

from peepholelib.training.save_fns import default_save
from peepholelib.training.load_fns import default_load
from peepholelib.training.accuracy_fns import img_classification_acc
from peepholelib.training.train_loops import default_train_loop
from peepholelib.training.val_loops import default_val_loop
from peepholelib.training.test_loops import default_test_loop

class Trainer():
    def __init__(self, **kwargs):
        """
        Base trainer that owns the full training state and orchestrates the
        training/validation/testing workflow via pluggable loop strategies.

        It builds dataloaders, tracks losses/accuracies, handles checkpoint
        resume, early stopping, and scheduler stepping. The actual per-epoch
        logic is delegated to `train_loop`, `val_loop`, and
        `save_fn`, which can be swapped to customize behavior without
        rewriting the whole trainer.

        Args:
        - model (peepholelib.models.model_wrap.modelWrap): model wrapper to train/evaluate.
        - path (str|pathlib.Path): directory where checkpoints and plots are stored.
        - name (str): base filename for checkpoints
        - verbose (bool): log progress to stdout

        - datasets (dict[str]: torch.utils.data.Dataset): dataset splits, e.g., `'train'`/`'val'`/`'test'`. A `DataLoader` is built once per key.
        - dataloader_kwargs (dict[str]: dict): kwargs for `torch.utils.data.DataLoader`, keyed by `dataset_key`, one dict per split (including `batch_size`, `shuffle`).
        - iterations (dict[str]: int|"full"): number of iterations per epoch, keyed by `dataset_key`. `"full"` iterates over the whole split.
        - in_parser (callable): function to map raw batch -> dict with tensors
        - out_parser (callable): function to map model output -> prediction tensor
        - loss_fn (callable)/loss_kwargs (dict): loss class and kwargs
        - acc_fn (callable): accuracy function
        - optimizer (torch.optim.<optimizer>): torch optimizer instance.
        - scheduler (torch.optim.lr_scheduler.<scheduler>): optional LR scheduler instance.
        - max_epochs (int): maximum number of epochs to run. Default to `1000`.
        - early_stopping_patience (int): stops the training if the validation loss does not improve for this amnount of epochs. Defaults to `inf`.
        - train_loop/val_loop/test_loop (callable): pluggable loop functions, default to `peepholelib.training.<train|val|test>_loops.default_<train|val|test>_loop`. Each accepts a `dataset_key` kwarg (default `'train'`/`'val'`/`'test'`) selecting which entry of `self._dls`/`self._iters` to use.
        - save_fn/load_fn: pluggable checkpoint function. Examples at `peepholelib.training.<save|load>_fns.py`.
        - save_every (int): save the model every `save_every` training epochs. If `None` (Default) intermediate checkpoints are not saved.
        """

        # Preliminaries
        self.model = kwargs['model']
        self.device = self.model.device
        self.path = Path(kwargs['path'])
        self.name = kwargs['name']
        self.verbose = kwargs.get('verbose', False)

        self.in_parser = kwargs.get('in_parser', lambda x:x)
        self.out_parser = kwargs.get('out_parser', lambda x:x)

        # Dataloaders, built once per dataset_key
        datasets = kwargs['datasets']
        dataloader_kwargs = kwargs['dataloader_kwargs']
        iterations = kwargs['iterations']

        self._iters = {
                _key: (ceil(len(_ds)/dataloader_kwargs[_key]['batch_size']) if iterations[_key] == 'full' else iterations[_key])
                for _key, _ds in datasets.items()
                }

        self._dls = {}
        for _key, _ds in datasets.items():
            self._dls[_key] = DataLoader(dataset=_ds, **dataloader_kwargs[_key])

        # Loss function
        _l = kwargs.get('loss_fn', torch.nn.CrossEntropyLoss)
        loss_kwargs = kwargs.get('loss_kwargs', dict(reduction='sum'))
        self.acc_fn = kwargs.get('acc_fn', img_classification_acc)

        self.loss_fn = _l(**loss_kwargs)

        # Training artifacts
        self.max_epochs = kwargs.get('max_epochs', 1000)
        self.optim = kwargs['optimizer']
        self.scheduler = kwargs.get('scheduler', None)

        # training functions
        self.train_loop = partial(
                kwargs.get("train_loop", default_train_loop),
                self = self
                )
        self.val_loop = partial(
                kwargs.get("val_loop", default_val_loop),
                self = self
                )
        self.test_loop = partial(
                kwargs.get("test_loop", default_test_loop),
                self = self
                )
        self.save_fn = partial(
                kwargs.get("save_fn", default_save),
                self = self
                )
        self.load_fn = partial(
                kwargs.get("load_fn", default_load),
                self = self
                )
        self.save_every = kwargs.get("save_every", None)
        self.early_stopping_patience = kwargs.get("early_stopping_patience", float('inf'))

        # set file names
        self.file = self.path/self.name
        self.best_model_file = self.path/'best_model'/(self.name+'.pt')

        # check if there are previous trainings and compute the phase num
        _prev_plots = list(self.path.glob(self.name+'.phase_*.losses.png'))
        if len(_prev_plots) > 0:
            _phase_nums = [int(re.search(r'phase_(\d+)', p.name)[1]) for p in _prev_plots]
            self.phase_num = max(_phase_nums) + 1 
        else:
            self.phase_num = 0
        self.loss_plot_file = self.path/(self.name+f'.phase_{self.phase_num}.losses.png')
        if self.verbose: print(f'Loss plot will be saved as \'{self.loss_plot_file}\'.')

        # create dirs
        self.path.mkdir(parents=True, exist_ok=True)
        self.best_model_file.parent.mkdir(parents=True, exist_ok=True)

        # Pre-allocate training history buffers
        self.train_losses = torch.zeros(self.max_epochs, requires_grad=False)
        self.val_losses = torch.zeros(self.max_epochs, requires_grad=False)
        self.train_acc = torch.zeros(self.max_epochs, requires_grad=False)
        self.val_acc = torch.zeros(self.max_epochs, requires_grad=False)

        # Check model existance
        if self.best_model_file.exists():
            if self.verbose: print(f'Found best_model file {self.best_model_file.as_posix()}. Resume training')
            self.load_fn(file = self.best_model_file)
            
        else:
            if self.verbose: print('No training ongoing, starting anew.')
            self.initial_epoch = 0
            self.best_val_loss = float('inf')
            self.best_epoch = 0

        self.num_bad_epochs = 0
        return
   
    def _train_epoch(self, epoch):
        t0 = time()
        self.train_loop(epoch=epoch)
        stop = self.val_loop(epoch=epoch)

        if self.save_every != None and self.save_every != 0:
            if (epoch + 1) % self.save_every == 0:
                self.save_fn(
                        epoch = epoch,
                        file = self.file.as_posix()+f'.{epoch}.pt',
                        plot = True
                        )

        if self.verbose: 
            print(
                f'epoch {epoch} - train loss: {self.train_losses[epoch]:.4f} - '
                f'val loss: {self.val_losses[epoch]:.4f} - '
                f'train acc: {self.train_acc[epoch]*100:.2f} - '
                f'val acc: {self.val_acc[epoch]*100:.2f} - '
                f'time: {time()-t0:.2f}'
            )

        return stop 

    def fit(self):
        if self.initial_epoch >= self.max_epochs:
            print(f'Already trained for {self.initial_epoch} epochs, not training.')
            return

        if self.verbose: print('----- Training Model ----- ')

        for epoch in range(self.initial_epoch, self.max_epochs):
            stop = self._train_epoch(epoch)
            if stop: break

        return
                
    def test(self):
        if self.verbose: print('----- Testing Model ----- ')

        if self.verbose: print(f'Loading best model config from {self.best_model_file.as_posix()}')
        self.model.load_checkpoint(
                path = self.best_model_file.parent,
                name = self.best_model_file.name,
                verbose = self.verbose,
                )

        return self.test_loop()

    
