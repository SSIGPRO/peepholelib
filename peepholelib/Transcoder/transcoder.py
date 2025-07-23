import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from datetime import datetime

class FullyConnectedAutoencoder(nn.Module):
    def __init__(self, 
                 input_size,
                 embedding_size,
                 encoder_layer_dims=None,
                 decoder_layer_dims=None,
                 output_size=None,
                 activation='relu'):
        """
        Flexible Fully Connected Autoencoder
        
        Args:
            input_size (int): Size of the input vector
            embedding_size (int): Size of the embedding (bottleneck) layer
            encoder_layer_dims (list, optional): List of dimensions for encoder layers (excluding input and embedding).
                                                If None, creates a single layer encoder
            decoder_layer_dims (list, optional): List of dimensions for decoder layers (excluding embedding and output).
                                                If None, uses reversed encoder structure
            output_size (int, optional): Size of the output vector. If None, uses input_size
            activation (str): Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
        
        Example:
            # Simple autoencoder: 784 -> 128 -> 64 (embedding) -> 128 -> 784
            model = FlexibleFullyConnectedAutoencoder(
                input_size=784, 
                embedding_size=64, 
                encoder_layer_dims=[128]
            )
            
            # Complex autoencoder: 1000 -> 512 -> 256 -> 128 -> 32 (embedding) -> 64 -> 128 -> 256 -> 500
            model = FlexibleFullyConnectedAutoencoder(
                input_size=1000,
                embedding_size=32,
                encoder_layer_dims=[512, 256, 128],
                decoder_layer_dims=[64, 128, 256],
                output_size=500
            )
        """
        super(FullyConnectedAutoencoder, self).__init__()
        
        # Set default values
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.output_size = output_size if output_size is not None else input_size
        self.encoder_layer_dims = encoder_layer_dims if encoder_layer_dims is not None else []
        
        # If decoder_layer_dims not provided, use reversed encoder structure
        if decoder_layer_dims is None:
            self.decoder_layer_dims = list(reversed(self.encoder_layer_dims))
        else:
            self.decoder_layer_dims = decoder_layer_dims
            
        # Set activation function
        self.activation = self._get_activation(activation)
        
        # Build encoder and decoder
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
    def _get_activation(self, activation):
        """Get activation function"""
        activations = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': F.leaky_relu
        }
        return activations.get(activation.lower(), F.relu)
    
    def _get_activation_layer(self):
        """Get activation layer for nn.Sequential"""
        if self.activation == F.relu:
            return nn.ReLU()
        elif self.activation == torch.tanh:
            return nn.Tanh()
        elif self.activation == torch.sigmoid:
            return nn.Sigmoid()
        elif self.activation == F.leaky_relu:
            return nn.LeakyReLU()
        else:
            return nn.ReLU()
    
    def _build_encoder(self):
        """Build encoder layers"""
        layers = []
        
        # Create the complete layer size sequence: input -> hidden layers -> embedding
        layer_sizes = [self.input_size] + self.encoder_layer_dims + [self.embedding_size]
        
        # Build layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add activation except for the last layer (embedding layer)
            if i < len(layer_sizes) - 2:
                layers.append(self._get_activation_layer())
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        """Build decoder layers"""
        layers = []
        
        # Create the complete layer size sequence: embedding -> hidden layers -> output
        layer_sizes = [self.embedding_size] + self.decoder_layer_dims + [self.output_size]
        
        # Build layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add activation except for the last layer (output layer)
            if i < len(layer_sizes) - 2:
                layers.append(self._get_activation_layer())
        
        return nn.Sequential(*layers)
    
    def encode(self, x):
        """Encode input to embedding"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode embedding to output"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass: encode then decode"""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction, embedding
    
    def get_embedding(self, x):
        """Get embedding for input x"""
        with torch.no_grad():
            return self.encode(x)
    
    def print_architecture(self):
        """Print the architecture details"""
        encoder_sizes = [self.input_size] + self.encoder_layer_dims + [self.embedding_size]
        decoder_sizes = [self.embedding_size] + self.decoder_layer_dims + [self.output_size]
        
        print(f"Flexible Autoencoder Architecture:")
        print(f"Input size: {self.input_size}")
        print(f"Output size: {self.output_size}")
        print(f"Embedding size: {self.embedding_size}")
        print(f"Encoder layer dimensions: {self.encoder_layer_dims}")
        print(f"Decoder layer dimensions: {self.decoder_layer_dims}")
        
        print(f"\nEncoder architecture: {' -> '.join(map(str, encoder_sizes))}")
        print(f"Decoder architecture: {' -> '.join(map(str, decoder_sizes))}")
        
        print(f"\nEncoder layers:")
        for i, layer in enumerate(self.encoder):
            print(f"  {i}: {layer}")
        print(f"\nDecoder layers:")
        for i, layer in enumerate(self.decoder):
            print(f"  {i}: {layer}")
    
class AutoencoderTrainer:
    def __init__(self, model, path, train_dataset, val_dataset, test_dataset, 
                 batch_size=32, learning_rate=0.001, device=None):
        """
        Complete training setup for autoencoder
        
        Args:
            model: Autoencoder model
            train_data: Training dataset (torch.Tensor)
            val_data: Validation dataset (torch.Tensor)  
            test_data: Test dataset (torch.Tensor)
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.batch_size = batch_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.save_path = path
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Training setup
        self.optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=learning_rate,
                                    betas=(0.5, 0.999),
                                    eps=1e-7,
                                    weight_decay=1e-5,
                                    amsgrad=True)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Tracking variables
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_model_path = None
        
        # Create directory for saving models
        self.save_name = f"autoencoder_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = os.path.join(self.save_path, self.save_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"Training setup complete. Using device: {self.device}")
        print(f"Models will be saved in: {self.save_dir}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_data, batch_targets in self.train_loader:
            batch_data = batch_data.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            self.optimizer.zero_grad()
            reconstruction, embedding = self.model(batch_data)
            recon_loss = self.criterion(reconstruction, batch_targets)
            reg_loss = torch.mean(torch.sum(torch.abs(embedding), dim=1))
            #reg_loss  = torch.mean(torch.sum(embedding**2, dim=1))
            loss = recon_loss + 1e-1 * reg_loss
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data, batch_targets in self.val_loader:
                batch_data = batch_data.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                reconstruction, embedding = self.model(batch_data)
                recon_loss = self.criterion(reconstruction, batch_targets)
                reg_loss = torch.mean(torch.sum(torch.abs(embedding), dim=1))
                #reg_loss  = torch.mean(torch.sum(embedding**2, dim=1))
                loss = recon_loss + 1e-1 * reg_loss
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_best_model(self, epoch, val_loss):
        """Save model if it's the best so far"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            
            # Save model
            model_path = os.path.join(self.save_dir, f'best_model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': self.train_losses[-1],
                'val_loss': val_loss,
                'model_config': {
                    'input_size': self.model.input_size,
                    'embedding_size': self.model.embedding_size,
                    'output_size': self.model.output_size,
                    'encoder_layers': self.model.encoder_layer_dims,
                    'decoder_layers': self.model.decoder_layer_dims
                }
            }, model_path)
            
            # Remove previous best model
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
            
            self.best_model_path = model_path
            print(f"New best model saved! Epoch {epoch}, Val Loss: {val_loss:.6f}")
    
    def plot_losses(self, epoch):
        """Plot and save training/validation loss trends"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, 
                'b-', label='Training Loss', linewidth=2)
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, 
                'r-', label='Validation Loss', linewidth=2)
        
        # Mark best epoch
        plt.axvline(x=self.best_epoch + 1, color='green', linestyle='--', 
                   label=f'Best Epoch ({self.best_epoch + 1})')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss - Epoch {epoch + 1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'loss_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self, epochs, plot_every=5, early_stopping_patience=20):
        """
        Complete training loop
        
        Args:
            epochs: Number of epochs to train
            plot_every: Plot losses every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
        """
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Plotting every {plot_every} epochs")
        
        epochs_without_improvement = 0
        old_lr = self.optimizer.param_groups[0]['lr']
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            print(self.optimizer.param_groups[0]['lr'])
            if self.optimizer.param_groups[0]['lr'] != old_lr:
                print(f"Learning rate changed from {old_lr:.6f} to {self.optimizer.param_groups[0]['lr']:.6f}")
                old_lr = self.optimizer.param_groups[0]['lr']
            
            # Track losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.save_best_model(epoch, val_loss)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Print progress
            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Best Val: {self.best_val_loss:.6f} (Epoch {self.best_epoch + 1})")
            
            # Plot losses
            if (epoch + 1) % plot_every == 0:
                self.plot_losses(epoch)
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs.")
                break
        
        # Final plot
        if (epoch + 1) % plot_every != 0:
            self.plot_losses(epoch)
        
        print(f"\nTraining completed!")
        print(f"Best model: Epoch {self.best_epoch + 1} with validation loss: {self.best_val_loss:.6f}")
        print(f"Best model saved at: {self.best_model_path}")
    
    def load_best_model(self):
        """Load the best saved model"""
        if self.best_model_path and os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
            return checkpoint
        else:
            print("No best model found!")
            return None
    
    def evaluate_on_test(self):
        """Evaluate the model on test set and return detailed metrics"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_mae = 0
        num_samples = 0
        all_errors = []
        
        with torch.no_grad():
            for batch_data, batch_targets in self.test_loader:
                batch_data = batch_data.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                reconstruction, embedding = self.model(batch_data)
                
                # Calculate losses
                mse_loss = self.criterion(reconstruction, batch_targets)
                mae_loss = F.l1_loss(reconstruction, batch_targets)
                
                # Per-sample reconstruction errors
                sample_errors = torch.mean((reconstruction - batch_targets) ** 2, dim=1)
                all_errors.extend(sample_errors.cpu().numpy())
                
                total_loss += mse_loss.item() * batch_data.size(0)
                total_mse += mse_loss.item() * batch_data.size(0)
                total_mae += mae_loss.item() * batch_data.size(0)
                num_samples += batch_data.size(0)
        
        # Calculate final metrics
        mean_mse = total_mse / num_samples
        mean_mae = total_mae / num_samples
        std_error = np.std(all_errors)
        
        print(f"\n=== TEST SET EVALUATION ===")
        print(f"Mean Reconstruction Error (MSE): {mean_mse:.6f}")
        print(f"Mean Absolute Error (MAE): {mean_mae:.6f}")
        print(f"Standard Deviation of Errors: {std_error:.6f}")
        print(f"Min Error: {np.min(all_errors):.6f}")
        print(f"Max Error: {np.max(all_errors):.6f}")
        print(f"Number of test samples: {num_samples}")
        
        # Plot error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(all_errors, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(mean_mse, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean MSE: {mean_mse:.6f}')
        plt.xlabel('Reconstruction Error (MSE per sample)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reconstruction Errors on Test Set')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save error distribution plot
        error_plot_path = os.path.join(self.save_dir, 'test_error_distribution.png')
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        return {
            'mean_mse': mean_mse,
            'mean_mae': mean_mae,
            'std_error': std_error,
            'min_error': np.min(all_errors),
            'max_error': np.max(all_errors),
            'all_errors': all_errors
        }