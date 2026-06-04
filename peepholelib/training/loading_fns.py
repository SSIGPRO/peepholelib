import torch

def default_load(self, **kwargs):
    file = kwargs['file']

    data = torch.load(file, weights_only=False) 
    
    # to save accuracies and losses
    saved_len = len(data['train_losses'])
    self.train_losses[:saved_len] = data['train_losses']
    self.val_losses[:saved_len] = data['val_losses'] 
    self.train_acc[:saved_len] = data['train_accuracy']
    self.val_acc[:saved_len] = data['val_accuracy'] 
    self.best_epoch = data['best_epoch']
    self.best_val_loss = data['best_val_loss']

    self.model.load_checkpoint(
            path = file.parent,
            name = file.name,
            verbose = self.verbose
            )
    
    # resume from the checkpoint we loaded
    self.initial_epoch = self.best_epoch
    
    try:
        self.optim.load_state_dict(data['optimizer'])
    except (ValueError, KeyError):
        if self.verbose: print('Optimizer state incompatible with checkpoint, starting fresh.')
                                                                                                    
    if self.scheduler is not None and 'scheduler' in data:
        try:
            self.scheduler.load_state_dict(data['scheduler'])
        except (ValueError, KeyError):
            if self.verbose: print('Scheduler state incompatible with checkpoint, starting fresh.')
   return 
