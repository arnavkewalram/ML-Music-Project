"""
Training Pipeline Module for Music Transcription System

This module provides functionality for training the music transcription model.
It includes the Trainer class for model training, evaluation, and checkpointing.

Classes:
    Trainer: Handles model training, validation, and checkpointing
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.architecture import MusicTranscriptionModel
from model.losses import combined_loss


class Trainer:
    """
    Class for training the music transcription model.
    
    Attributes:
        model (nn.Module): Music transcription model
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        optimizer (optim.Optimizer): Optimizer for model training
        scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler
        device (torch.device): Device to use for training
        checkpoint_dir (str): Directory to save checkpoints
        log_dir (str): Directory to save logs
        loss_weights (Dict[str, float]): Weights for each loss component
        best_val_loss (float): Best validation loss
        patience (int): Patience for early stopping
        patience_counter (int): Counter for early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs',
        loss_weights: Optional[Dict[str, float]] = None,
        patience: int = 10
    ):
        """
        Initialize the Trainer with specified parameters.
        
        Args:
            model (nn.Module): Music transcription model
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader): DataLoader for validation data
            optimizer (optim.Optimizer): Optimizer for model training
            scheduler (optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
            device (torch.device, optional): Device to use for training
            checkpoint_dir (str): Directory to save checkpoints
            log_dir (str): Directory to save logs
            loss_weights (Dict[str, float], optional): Weights for each loss component
            patience (int): Patience for early stopping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.loss_weights = loss_weights
        self.patience = patience
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.component_losses = {
            'train': {
                'onset': [], 'pitch': [], 'duration': [], 'tempo': [],
                'time_signature': [], 'instrument': [], 'velocity': []
            },
            'val': {
                'onset': [], 'pitch': [], 'duration': [], 'tempo': [],
                'time_signature': [], 'instrument': [], 'velocity': []
            }
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Returns:
            Dict[str, float]: Dictionary of training losses
        """
        self.model.train()
        epoch_loss = 0.0
        component_losses = {
            'onset': 0.0, 'pitch': 0.0, 'duration': 0.0, 'tempo': 0.0,
            'time_signature': 0.0, 'instrument': 0.0, 'velocity': 0.0
        }
        num_batches = len(self.train_loader)
        
        # Progress bar
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Extract features and targets
            features = batch['melspectrogram']
            
            targets = {
                'onset_targets': batch['onset_target'],
                'pitch_targets': batch['pitch_target'],
                'duration_targets': batch['duration_target'],
                'velocity_targets': batch['velocity_target']
            }
            
            # Forward pass
            predictions = self.model(features)
            
            # Calculate loss
            loss, losses = combined_loss(predictions, targets, self.loss_weights)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            for key, value in losses.items():
                if key in component_losses:
                    component_losses[key] += value.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average losses
        epoch_loss /= num_batches
        for key in component_losses:
            component_losses[key] /= num_batches
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {'total': epoch_loss, **component_losses}
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model on the validation set.
        
        Returns:
            Dict[str, float]: Dictionary of validation losses
        """
        self.model.eval()
        epoch_loss = 0.0
        component_losses = {
            'onset': 0.0, 'pitch': 0.0, 'duration': 0.0, 'tempo': 0.0,
            'time_signature': 0.0, 'instrument': 0.0, 'velocity': 0.0
        }
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            # Progress bar
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Extract features and targets
                features = batch['melspectrogram']
                
                targets = {
                    'onset_targets': batch['onset_target'],
                    'pitch_targets': batch['pitch_target'],
                    'duration_targets': batch['duration_target'],
                    'velocity_targets': batch['velocity_target']
                }
                
                # Forward pass
                predictions = self.model(features)
                
                # Calculate loss
                loss, losses = combined_loss(predictions, targets, self.loss_weights)
                
                # Update metrics
                epoch_loss += loss.item()
                for key, value in losses.items():
                    if key in component_losses:
                        component_losses[key] += value.item()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average losses
        epoch_loss /= num_batches
        for key in component_losses:
            component_losses[key] /= num_batches
        
        return {'total': epoch_loss, **component_losses}
    
    def train(self, num_epochs: int, save_every: int = 1) -> Dict[str, List[float]]:
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs (int): Number of epochs to train
            save_every (int): Save checkpoint every n epochs
            
        Returns:
            Dict[str, List[float]]: Dictionary of training and validation losses
        """
        print(f"Training on device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Update tracking variables
            self.train_losses.append(train_losses['total'])
            self.val_losses.append(val_losses['total'])
            
            if self.scheduler is not None:
                self.learning_rates.append(self.scheduler.get_last_lr()[0])
            else:
                self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            for key in train_losses:
                if key != 'total' and key in self.component_losses['train']:
                    self.component_losses['train'][key].append(train_losses[key])
            
            for key in val_losses:
                if key != 'total' and key in self.component_losses['val']:
                    self.component_losses['val'][key].append(val_losses[key])
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch}/{num_epochs} - {epoch_time:.2f}s - "
                  f"Train Loss: {train_losses['total']:.4f} - "
                  f"Val Loss: {val_losses['total']:.4f}")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"New best model saved! Val Loss: {val_losses['total']:.4f}")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/{self.patience}")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping after {epoch} epochs")
                break
            
            # Save training curves
            self.plot_training_curves(epoch)
            
            # Save logs
            self.save_logs()
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'component_losses': self.component_losses
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            
        Returns:
            int: Epoch number of the checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Restore training history if available
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        if 'learning_rates' in checkpoint:
            self.learning_rates = checkpoint['learning_rates']
        
        if 'component_losses' in checkpoint:
            self.component_losses = checkpoint['component_losses']
        
        return checkpoint['epoch']
    
    def plot_training_curves(self, epoch: int) -> None:
        """
        Plot training and validation loss curves.
        
        Args:
            epoch (int): Current epoch
        """
        plt.figure(figsize=(15, 10))
        
        # Plot total losses
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 2, 2)
        plt.plot(self.learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate')
        plt.grid(True)
        
        # Plot component train losses
        plt.subplot(2, 2, 3)
        for key, values in self.component_losses['train'].items():
            if values:  # Only plot if there are values
                plt.plot(values, label=key)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train Component Losses')
        plt.legend()
        plt.grid(True)
        
        # Plot component validation losses
        plt.subplot(2, 2, 4)
        for key, values in self.component_losses['val'].items():
            if values:  # Only plot if there are values
                plt.plot(values, label=key)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Component Losses')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"training_curves_epoch_{epoch}.png"))
        plt.close()
    
    def save_logs(self) -> None:
        """Save training logs to JSON file."""
        logs = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'component_losses': self.component_losses,
            'best_val_loss': self.best_val_loss
        }
        
        with open(os.path.join(self.log_dir, "training_logs.json"), 'w') as f:
            json.dump(logs, f, indent=2)


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adam',
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    **kwargs
) -> optim.Optimizer:
    """
    Create optimizer for model training.
    
    Args:
        model (nn.Module): Model to optimize
        optimizer_type (str): Type of optimizer ('adam', 'sgd', 'adamw')
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for regularization
        **kwargs: Additional optimizer parameters
        
    Returns:
        optim.Optimizer: Optimizer instance
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'cosine',
    **kwargs
) -> optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer (optim.Optimizer): Optimizer
        scheduler_type (str): Type of scheduler ('step', 'cosine', 'plateau')
        **kwargs: Additional scheduler parameters
        
    Returns:
        optim.lr_scheduler._LRScheduler: Scheduler instance
    """
    if scheduler_type.lower() == 'step':
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_type.lower() == 'cosine':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
    elif scheduler_type.lower() == 'plateau':
        patience = kwargs.get('patience', 5)
        factor = kwargs.get('factor', 0.1)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=patience,
            factor=factor
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    optimizer_type: str = 'adam',
    scheduler_type: str = 'cosine',
    device: torch.device = None,
    checkpoint_dir: str = './checkpoints',
    log_dir: str = './logs',
    loss_weights: Optional[Dict[str, float]] = None,
    patience: int = 10,
    save_every: int = 5,
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train the music transcription model.
    
    Args:
        model (nn.Module): Music transcription model
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for regularization
        optimizer_type (str): Type of optimizer ('adam', 'sgd', 'adamw')
        scheduler_type (str): Type of scheduler ('step', 'cosine', 'plateau')
        device (torch.device, optional): Device to use for training
        checkpoint_dir (str): Directory to save checkpoints
        log_dir (str): Directory to save logs
        loss_weights (Dict[str, float], optional): Weights for each loss component
        patience (int): Patience for early stopping
        save_every (int): Save checkpoint every n epochs
        resume_from (str, optional): Path to checkpoint to resume from
        
    Returns:
        Dict[str, Any]: Dictionary of training results
    """
    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=scheduler_type,
        T_max=num_epochs
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        loss_weights=loss_weights,
        patience=patience
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if resume_from is not None:
        start_epoch = trainer.load_checkpoint(resume_from) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Train model
    results = trainer.train(num_epochs - start_epoch + 1, save_every=save_every)
    
    return {
        'model': model,
        'trainer': trainer,
        'results': results
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    from preprocessing.dataset_preprocessor import create_datasets, create_train_transform, create_validation_transform
    
    parser = argparse.ArgumentParser(description="Train music transcription model")
    parser.add_argument("--preprocessed_dir", type=str, required=True, help="Directory containing preprocessed data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="cosine", help="Scheduler type")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every n epochs")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for DataLoader")
    
    args = parser.parse_args()
    
    # Create datasets and data loaders
    transforms = {
        'train': create_train_transform(),
        'validation': create_validation_transform(),
        'test': create_validation_transform()
    }
    
    dataloaders = create_datasets(
        preprocessed_dir=args.preprocessed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transforms=transforms
    )
    
    # Create model
    model = MusicTranscriptionModel(
        input_channels=1,
        hidden_channels=[32, 64, 128, 256],
        lstm_hidden_size=256,
        attention_dim=128,
        num_pitches=128,
        max_duration_frames=100,
        num_instruments=10,
        dropout=0.2
    )
    
    # Train model
    results = train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['validation'],
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        patience=args.patience,
        save_every=args.save_every,
        resume_from=args.resume_from
    )
    
    print("Training completed!")
