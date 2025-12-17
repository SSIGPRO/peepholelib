import os
import torch.nn as nn
import torch
import numpy as np
import json
try:
    from tqdm.auto import tqdm
except ImportError:  # fallback for environments without tqdm.auto
    from tqdm import tqdm

import math

from collections import OrderedDict
# Dynamically import EarlyStopping from utils-2d.py (filename with hyphen)
import importlib.util as _importlib_util
from pathlib import Path
_UTILS2D_PATH = Path(__file__).parent / "utils2d.py"
_spec_es = _importlib_util.spec_from_file_location("utils_2d", _UTILS2D_PATH)
_utils2d_mod = _importlib_util.module_from_spec(_spec_es)
_spec_es.loader.exec_module(_utils2d_mod)  # type: ignore
EarlyStopping = _utils2d_mod.EarlyStopping


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
class Encoder2D(nn.Module):
    def __init__(self, num_sensors=19, seq_len=128, in_ch = 19, embedding_size='medium', lay3=False, kernel_size=[3,3]):
        super().__init__()
        self.num_sensors = num_sensors
        self.seq_len = seq_len
        self.lay3 = lay3
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.num_filter = max(19, 2*self.num_sensors)

        # padding automatico per "same"
        self.padding1 = (self.kernel_size[0]-1)//2
        self.padding2 = (self.kernel_size[1]-1)//2

        compression_map = {
            'xlarge': 0.5,
            'large': 1,
            'medium': 2,
            'small': 4,
            'xsmall': 8,
            '2xsmall': 16
        }
        compression_factor = compression_map[embedding_size]

        layers = OrderedDict()
        # Layer 1
        out1 = self.in_ch * 2
        layers['layer1'] = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, out1, kernel_size=(self.kernel_size[0], self.kernel_size[1]),
                                padding=(self.padding1, self.padding2))),
            ('batchnorm1', nn.BatchNorm2d(out1)),
            ('maxpool1', nn.MaxPool2d((1,2))),
            ('relu1', nn.ReLU())
        ]))
        # Layer 2
        out2 = out1 * 2
        layers['layer2'] = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(out1, out2, kernel_size=(self.kernel_size[0], self.kernel_size[1]),
                                padding=(self.padding1, self.padding2))),
            ('batchnorm2', nn.BatchNorm2d(out2)),
            ('maxpool2', nn.MaxPool2d((1,2))),
            ('relu2', nn.ReLU())
        ]))
        # Layer 3 opzionale
        if lay3:
            out3 = out2 * 2
            layers['layer3'] = nn.Sequential(OrderedDict([
                ('conv3', nn.Conv2d(out2, out3, kernel_size=(self.kernel_size[0], self.kernel_size[1]),
                                    padding=(self.padding1, self.padding2))),
                ('batchnorm3', nn.BatchNorm2d(out3)),
                ('maxpool3', nn.MaxPool2d((1,2))),
                ('relu3', nn.ReLU())
            ]))
            final_out = out3
        else:
            final_out = out2

        self.nn_enc_body = nn.Sequential(layers)
        self.flatten = nn.Flatten()

        # Capture encoder output geometry for decoder
        (self.flattened_size,
         self.enc_out_channels,
         self.enc_out_height,
         self.enc_out_width) = self.compute_flattened_dim()
        self.seq_len_out = self.enc_out_width
        self.latent_dim = int((self.num_filter*self.num_sensors)/compression_factor)
        self.linear = nn.Linear(self.flattened_size, self.latent_dim)

    def forward(self, x):
        for name, block in self.nn_enc_body.named_children():
            if isinstance(block, nn.Sequential):
                for sub_name, layer in block.named_children():
                    x = layer(x)
                    # print(f"Encoder2D.{name}.{sub_name} → {tuple(x.shape)}")
            else:
                x = block(x)
                # print(f"Encoder2D.{name} → {tuple(x.shape)}")

        x = self.flatten(x)
        # print(f"Encoder2D.flatten → {tuple(x.shape)}")

        x = self.linear(x)
        # print(f"Encoder2D.linear → {tuple(x.shape)}")
        return x

    def compute_flattened_dim(self):
        with torch.no_grad():
            x = torch.zeros((1,1,self.num_sensors,self.seq_len))
            x = self.nn_enc_body(x)
            _, c, h, w = x.size()
        return c*h*w, c, h, w
# ===========================
#       Decoder 2D
# ===========================
class Decoder2D(nn.Module):
    def __init__(self, num_sensors=19, seq_len=128, flattened_size=1024,
                 latent_dim=128, embedding_size='medium', lay3=False,
                 kernel_size=[3,5], encoder_shape=None):
        super().__init__()

        self.num_sensors = num_sensors
        self.seq_len = seq_len
        self.lay3 = lay3
        self.flattened_size = flattened_size
        self.latent_dim = latent_dim
       
        self.kernel_size = kernel_size

        # padding “same” per convoluzioni
        self.padding1 = (self.kernel_size[0]-1)//2
        self.padding2 = (self.kernel_size[1]-1)//2
        
        layers = OrderedDict()

        # ────────── Linear + Reshape (collo di bottiglia) ──────────
        layers['linear'] = nn.Linear(self.latent_dim, self.flattened_size)
        if encoder_shape is not None:
            c, h, w_enc = encoder_shape
        else:
            # fall back to previous heuristic if shape is not provided
            c = self.num_sensors * 8 if lay3 else self.num_sensors * 4
            h = self.num_sensors
            w_enc = max(1, self.seq_len // (8 if lay3 else 4))

        self._enc_channels = c
        self._enc_height = h
        self._enc_width = w_enc

        layers['reshape'] = Reshape((self._enc_channels, self._enc_height, self._enc_width))

        # ────────── Deconv Layer 1 ──────────
        out1 = max(self._enc_channels // 2, 1)
        layers['deconv1'] = nn.Sequential(OrderedDict([
            ('conv_transpose1', nn.ConvTranspose2d(in_channels=self._enc_channels, out_channels=out1,
                                                    kernel_size=(1, 2),
                                                    stride=(1,2),
                                                    )),
            ('conv1', nn.Conv2d(out1, out1,
                                kernel_size=(self.kernel_size[0], self.kernel_size[1]),
                                padding=(self.padding1, self.padding2))),
            ('relu1', nn.ReLU()),
            ('batchnorm1', nn.BatchNorm2d(out1))
        ]))

        # ────────── Deconv Layer 2 ──────────
        out2 = max(out1 // 2, 1)
        layers['deconv2'] = nn.Sequential(OrderedDict([
            ('conv_transpose2', nn.ConvTranspose2d(in_channels=out1, out_channels=out2,
                                                    kernel_size=(1,2),
                                                    stride=(1,2),
                                                    )),
            ('conv2', nn.Conv2d(out2, out2,
                                kernel_size=(self.kernel_size[0], self.kernel_size[1]),
                                padding=(self.padding1, self.padding2))),
            ('relu2', nn.ReLU()),
            ('batchnorm2', nn.BatchNorm2d(out2))
        ]))
        last_out = out2

        # ────────── Deconv Layer 3 opzionale ──────────
        if lay3:
            out3 = max(out2 // 2, 1)
            last_out = out3
            layers['deconv3'] = nn.Sequential(OrderedDict([
                ('conv_transpose3', nn.ConvTranspose2d(in_channels=out2, out_channels=out3,
                                                        kernel_size=(1, 2),
                                                        stride=(1,2),
                                                        )),
                ('conv3', nn.Conv2d(out3, out3,
                                    kernel_size=(self.kernel_size[0], self.kernel_size[1]),
                                    padding=(self.padding1, self.padding2))),
                ('relu3', nn.ReLU()),
                ('batchnorm3', nn.BatchNorm2d(out3))
            ]))

        # ────────── Ultimo layer: ricostruzione canale 1 ──────────
        layers['last_conv'] = nn.Conv2d(last_out, 1, kernel_size=1)

        self.nn_dec_body = nn.Sequential(layers)

    def forward(self, x):
        for name, block in self.nn_dec_body.named_children():
            # Se il blocco è una sequenza di layer
            if isinstance(block, nn.Sequential):
                for sub_name, layer in block.named_children():
                    x = layer(x)
                    # print(f"Decoder2D.{name}.{sub_name} → {tuple(x.shape)}")
            else:
                # Linear, Reshape, ecc. passano di qui
                x = block(x)
                # print(f"Decoder2D.{name} → {tuple(x.shape)}")
        return x

##############################
#
# define the NN architecture
#
###############################

class CONV_AE2D(nn.Module):
    
    def __init__(self, num_sensors=19, seq_len=128, kernel_size=[3,5],
                 embedding_size='medium', lay3=False):
        super(CONV_AE2D, self).__init__()

        self.num_sensors = num_sensors
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.embedding_size = embedding_size
        self.lay3 = lay3

        # Encoder
        self.encoder = Encoder2D(
            num_sensors=self.num_sensors,
            seq_len=self.seq_len,
            kernel_size=self.kernel_size,   ## <--- ho aggiunto questo. Francesco
            embedding_size=self.embedding_size,
            lay3=self.lay3
        )
        
        # Recupera dimensioni per il decoder
        self.flattened_size = self.encoder.flattened_size
        self.latent_dim = self.encoder.latent_dim
        self.num_filters = self.encoder.num_filter

        # Decoder
        self.decoder = Decoder2D(
            num_sensors=self.num_sensors,
            seq_len=self.seq_len,
            flattened_size=self.flattened_size,
            latent_dim=self.latent_dim,
            embedding_size=self.embedding_size,
            lay3=self.lay3,
            kernel_size=self.kernel_size,
            encoder_shape=(self.encoder.enc_out_channels,
                           self.encoder.enc_out_height,
                           self.encoder.enc_out_width)
        )

    def forward(self, x):
        # x shape: [B, 1, num_sensors, seq_len]
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out, enc

#################################


def train_conv_ae2D(train_iter, test_iter, model, criterion, 
                    optimizer, scheduler, device, out_dir, model_name, 
                        epochs=100, es_patience=10, seed=None,k = None, writer = None):
        """
        Training function for the 1D Convolutional AutoEncoder model.

        Args:
            param_conf (dict): Configuration parameters.
            train_iter (DataLoader): Iterator for the training data.
            test_iter (DataLoader): Iterator for the test data.
            model (torch.nn.Module): The model to train.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            device (torch.device): Device for training (CPU or GPU).
            out_dir (str): Directory to save the trained models.
            model_name (str): Name of the model.
            epochs (int): Number of epochs to train.
            es_patience (int): Number of epochs without improvement before early stopping.
            seed (int): Random seed for reproducibility.

        Returns:
            model (torch.nn.Module): The trained model.
        """
        # Create the output directory if it doesn't exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        # Setup early stopping to monitor validation loss
        early_stopping = EarlyStopping(patience=es_patience)

        val_loss = 10 ** 16  # Initialize a large value for the validation loss
        model.to(device)  # Move model to the appropriate device (CPU/GPU)
        prev_lr = optimizer.param_groups[0]['lr']
        best_train_loss = float('inf')

        # Start the training loop
        for epoch in tqdm(range(epochs), unit='epoch', dynamic_ncols=True):
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f" len of test iter: {len(train_iter)}")
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_steps = 0
            for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), unit="batch", dynamic_ncols=True):
                # Skip empty or invalid batches
                # if batch.ndim == 1 or batch.shape[0] == 0:
                #     continue
                

                inputs, targets = batch['data'].to(device), batch['label'].to(device)
                inputs, targets = inputs[:, 0].to(device), targets[:, 0].to(device)
                # inputs = batch[:, 0].to(device)  # Get inputs from the batch
                # targets = batch[:, 1].to(device)  # Get targets from the batch
                
                # Permute the inputs and targets if necessary (to match expected shape)
                inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # → [B, 1, F, T]
                targets = targets.permute(0, 2, 1).unsqueeze(1)  # → [B, 1, F, T]
                inputs.to(device)
                targets.to(device)
                optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                outputs = model(inputs).to(device)
                loss = criterion(outputs, targets)  # Calculate the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update model weights

                train_loss += loss.item()  # Accumulate training loss
                train_steps += 1
                
            # Compute and print the average training loss
            avg_train_loss = train_loss / train_steps
            print(f"Train loss: {avg_train_loss:.4f}")
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
           
            # Validation phase
            model.eval()  # Set model to evaluation mode
            val_loss_temp = 0.0
            val_steps = 0
            with torch.no_grad():  # Disable gradient computation for validation
                for i, batch in tqdm(enumerate(test_iter), total=len(test_iter), desc="Evaluating", dynamic_ncols=True):
                    # Skip empty or invalid batches
                    # if batch.ndim == 1 or batch.shape[0] == 0:
                    #     continue

                    inputs, targets = batch['data'].to(device), batch['label'].to(device)
                    inputs, targets = inputs[:, 0], targets[:, 0]
                    # inputs = batch[:, 0].to(device)
                    # targets = batch[:, 1].to(device)

                    # Permute inputs and targets
                    inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # → [B, 1, F, T]
                    targets = targets.permute(0, 2, 1).unsqueeze(1)  # → [B, 1, F, T]

                    # Forward pass for validation
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)  # Calculate validation loss
                    val_loss_temp += loss.item()  # Accumulate validation loss
                    val_steps += 1

            # Compute and print the average validation loss
            avg_val_loss = val_loss_temp / val_steps
            print(f"Validation loss: {avg_val_loss:.4f}")
            # Log su TensorBoard
            if writer is not None:
                writer.add_scalars('Loss', {  # nota la S finale: add_scalars
                                    'Train': avg_train_loss,
                                    'Validation': avg_val_loss
                                }, epoch)
                lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning Rate', lr, epoch)

            # Early stopping: if validation loss doesn't improve, stop training
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

            # Update the learning rate scheduler based on validation loss
            scheduler.step(avg_val_loss)
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]['lr']
            if current_lr != prev_lr:
                print(f"LR updated: {prev_lr:.3e} → {current_lr:.3e}")
                prev_lr = current_lr

            # If the validation loss has improved, save the model checkpoint
            if avg_val_loss < val_loss:
                print(f"Validation loss improved from {val_loss:.4f} to {avg_val_loss:.4f}, saving model.")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'seed': seed,
                    'val_loss': avg_val_loss
                }, os.path.join(out_dir, f'{model_name}.pth'))
                val_loss = avg_val_loss
        if writer is not None:
            writer.flush()
        
        model_info = {}
        try:
            # Load existing model info if it exists
            model_data = {
                "Best Train Loss": best_train_loss,
                "Best Val Loss": val_loss,
                "Final Val Loss": avg_val_loss,
                "lay3": model.encoder.lay3,
                "Latent Dim": model.encoder.latent_dim,
                "kernel_size": model.encoder.kernel_size,
                "seql": model.encoder.seq_len,
                "in_c": model.encoder.num_sensors,
                "k": k,
                "seed": seed,
            }
            file_path = Path(out_dir) / "model_info.json"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists():
                with open(file_path, "r") as f:
                    model_info = json.load(f)

            model_info[model_name] = model_data

            with open(file_path, "w") as f:
                json.dump(model_info, f, indent=4)
        except Exception as e:
            print(f"Error during model info saving: {e}")
        print("Training complete.")
        if writer is not None:
            writer.close()
        return model

def salva_tensore(nome_file, blocco_nome, layer_nome, tensor):
    with open(nome_file, 'a') as f:
        f.write(f"\n--- {blocco_nome} → {layer_nome} ---\n")
        np_array = tensor.detach().cpu().numpy()
        np.set_printoptions(threshold=np.inf, linewidth=200)  # Nessun troncamento
        f.write(np.array2string(np_array, precision=4, separator=', '))

def build_model(seq_len: int, num_channels: int, kernel_size, emb_size: str,
                lay3: bool, device: torch.device):
    """Instantiate the 2D Conv AE with consistent shapes."""
    model = CONV_AE2D(
        num_sensors=num_channels,
        seq_len=seq_len,
        embedding_size=emb_size,
        lay3=lay3,
        kernel_size=kernel_size,
    ).to(device)
    return model