"""
Utility functions for the Music Transcription System.

This module provides various utility functions used across the system.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional, Union, Any


def create_directory_structure(base_dir: str) -> None:
    """
    Create the directory structure for the music transcription system.
    
    Args:
        base_dir (str): Base directory path
    """
    directories = [
        'model',
        'preprocessing',
        'inference',
        'utils',
        'scripts',
        'configs',
        'data',
        'data/maestro',
        'data/preprocessed',
        'output',
        'checkpoints',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)
    
    print(f"Directory structure created at {base_dir}")


def plot_waveform(
    audio_data: np.ndarray,
    sample_rate: int,
    title: str = "Waveform",
    figsize: Tuple[int, int] = (10, 4),
    output_path: Optional[str] = None
) -> None:
    """
    Plot waveform of audio data.
    
    Args:
        audio_data (np.ndarray): Audio time series
        sample_rate (int): Sample rate of audio
        title (str): Title of plot
        figsize (Tuple[int, int]): Figure size
        output_path (str, optional): Path to save plot
    """
    plt.figure(figsize=figsize)
    
    # Time axis
    time = np.arange(len(audio_data)) / sample_rate
    
    plt.plot(time, audio_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_spectrogram(
    spectrogram: np.ndarray,
    sample_rate: int,
    hop_length: int,
    title: str = "Spectrogram",
    figsize: Tuple[int, int] = (10, 4),
    output_path: Optional[str] = None
) -> None:
    """
    Plot spectrogram.
    
    Args:
        spectrogram (np.ndarray): Spectrogram data
        sample_rate (int): Sample rate of audio
        hop_length (int): Hop length used for STFT
        title (str): Title of plot
        figsize (Tuple[int, int]): Figure size
        output_path (str, optional): Path to save plot
    """
    plt.figure(figsize=figsize)
    
    # Time axis
    time_steps = spectrogram.shape[1]
    time = np.arange(time_steps) * hop_length / sample_rate
    
    # Frequency axis
    freq_bins = spectrogram.shape[0]
    freqs = np.arange(freq_bins) * sample_rate / (2 * freq_bins)
    
    plt.imshow(
        spectrogram,
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], freqs[0], freqs[-1]],
        cmap='viridis'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_mel_spectrogram(
    mel_spectrogram: np.ndarray,
    sample_rate: int,
    hop_length: int,
    title: str = "Mel Spectrogram",
    figsize: Tuple[int, int] = (10, 4),
    output_path: Optional[str] = None
) -> None:
    """
    Plot mel spectrogram.
    
    Args:
        mel_spectrogram (np.ndarray): Mel spectrogram data
        sample_rate (int): Sample rate of audio
        hop_length (int): Hop length used for STFT
        title (str): Title of plot
        figsize (Tuple[int, int]): Figure size
        output_path (str, optional): Path to save plot
    """
    plt.figure(figsize=figsize)
    
    # Time axis
    time_steps = mel_spectrogram.shape[1]
    time = np.arange(time_steps) * hop_length / sample_rate
    
    # Mel bins axis
    mel_bins = mel_spectrogram.shape[0]
    
    plt.imshow(
        mel_spectrogram,
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], 0, mel_bins],
        cmap='viridis'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Bins")
    plt.title(title)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        output_path (str): Path to save configuration
    """
    import json
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save configuration
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {output_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    import json
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in model.
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module) -> None:
    """
    Print summary of model architecture.
    
    Args:
        model (torch.nn.Module): PyTorch model
    """
    print(f"Model Architecture: {model.__class__.__name__}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print("Layer summary:")
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {module.__class__.__name__} ({params:,} parameters)")


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def get_available_device() -> torch.device:
    """
    Get available device (CUDA or CPU).
    
    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    return device


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Music Transcription System Utilities")
    parser.add_argument("--create_dirs", action="store_true", help="Create directory structure")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory path")
    
    args = parser.parse_args()
    
    if args.create_dirs:
        create_directory_structure(args.base_dir)
