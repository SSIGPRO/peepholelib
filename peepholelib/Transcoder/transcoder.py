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
                 output_size=None,
                 encoder_layers=3,
                 decoder_layers=None,
                 activation='relu'):
        """
        Fully Connected Autoencoder
        
        Args:
            input_size (int): Size of the input vector
            embedding_size (int): Size of the embedding (bottleneck) layer
            output_size (int, optional): Size of the output vector. If None, uses input_size
            encoder_layers (int): Number of layers in the encoder (default: 3)
            decoder_layers (int, optional): Number of layers in the decoder. If None, uses encoder_layers
            activation (str): Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
        """
        super(FullyConnectedAutoencoder, self).__init__()
        
        # Set default values
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.output_size = output_size if output_size is not None else input_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers if decoder_layers is not None else encoder_layers
        
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
    
    def _build_encoder(self):
        """Build encoder layers"""
        layers = []
        
        if self.encoder_layers == 1:
            # Direct connection to embedding
            layers.append(nn.Linear(self.input_size, self.embedding_size))
        else:
            # Calculate layer sizes - gradually reduce from input to embedding
            layer_sizes = self._calculate_layer_sizes(
                self.input_size, 
                self.embedding_size, 
                self.encoder_layers
            )
            
            # Build layers
            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                
                # Add activation except for the last layer (embedding layer)
                if i < len(layer_sizes) - 2:
                    layers.append(nn.ReLU() if self.activation == F.relu else 
                                nn.Tanh() if self.activation == torch.tanh else
                                nn.Sigmoid() if self.activation == torch.sigmoid else
                                nn.LeakyReLU())
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        """Build decoder layers"""
        layers = []
        
        if self.decoder_layers == 1:
            # Direct connection from embedding to output
            layers.append(nn.Linear(self.embedding_size, self.output_size))
        else:
            # Calculate layer sizes - gradually increase from embedding to output
            layer_sizes = self._calculate_layer_sizes(
                self.embedding_size, 
                self.output_size, 
                self.decoder_layers
            )
            
            # Build layers
            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                
                # Add activation except for the last layer (output layer)
                if i < len(layer_sizes) - 2:
                    layers.append(nn.ReLU() if self.activation == F.relu else 
                                nn.Tanh() if self.activation == torch.tanh else
                                nn.Sigmoid() if self.activation == torch.sigmoid else
                                nn.LeakyReLU())
        
        return nn.Sequential(*layers)
    
    def _calculate_layer_sizes(self, start_size, end_size, num_layers):
        """Calculate intermediate layer sizes"""
        if num_layers == 1:
            return [start_size, end_size]
        
        # Use geometric progression for smooth size transitions
        ratio = (end_size / start_size) ** (1 / (num_layers - 1))
        sizes = [start_size]
        
        for i in range(1, num_layers):
            size = int(start_size * (ratio ** i))
            # Ensure we don't go below end_size for encoder or above end_size for decoder
            if start_size > end_size:  # Encoder case
                size = max(size, end_size)
            else:  # Decoder case
                size = min(size, end_size)
            sizes.append(size)
        
        sizes[-1] = end_size  # Ensure exact end size
        return sizes
    
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
        print(f"Autoencoder Architecture:")
        print(f"Input size: {self.input_size}")
        print(f"Output size: {self.output_size}")
        print(f"Embedding size: {self.embedding_size}")
        print(f"Encoder layers: {self.encoder_layers}")
        print(f"Decoder layers: {self.decoder_layers}")
        print(f"\nEncoder:")
        for i, layer in enumerate(self.encoder):
            print(f"  {i}: {layer}")
        print(f"\nDecoder:")
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
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
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
            reconstruction, _ = self.model(batch_data)
            loss = self.criterion(reconstruction, batch_targets)
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
                
                reconstruction, _ = self.model(batch_data)
                loss = self.criterion(reconstruction, batch_targets)
                
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
                    'encoder_layers': self.model.encoder_layers,
                    'decoder_layers': self.model.decoder_layers
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