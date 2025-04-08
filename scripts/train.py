"""
Main script for training the music transcription model.

This script provides a command-line interface for training the model
with various configuration options.
"""

import os
import argparse
import torch
import json
from datetime import datetime

from model.architecture import MusicTranscriptionModel
from model.trainer import train_model
from preprocessing.dataset_preprocessor import (
    download_maestro, preprocess_maestro, create_datasets,
    create_train_transform, create_validation_transform, create_test_transform
)


def main():
    parser = argparse.ArgumentParser(description="Train music transcription model")
    
    # Dataset options
    parser.add_argument("--dataset_dir", type=str, default="./data/maestro", 
                        help="Directory to download/store MAESTRO dataset")
    parser.add_argument("--preprocessed_dir", type=str, default="./data/preprocessed", 
                        help="Directory to store preprocessed data")
    parser.add_argument("--download", action="store_true", 
                        help="Download MAESTRO dataset")
    parser.add_argument("--preprocess", action="store_true", 
                        help="Preprocess MAESTRO dataset")
    parser.add_argument("--max_items", type=int, default=None, 
                        help="Maximum number of items to process per split")
    
    # Audio processing options
    parser.add_argument("--sample_rate", type=int, default=22050, 
                        help="Sample rate for audio processing")
    parser.add_argument("--n_fft", type=int, default=2048, 
                        help="FFT window size")
    parser.add_argument("--hop_length", type=int, default=512, 
                        help="Hop length for STFT")
    parser.add_argument("--n_mels", type=int, default=128, 
                        help="Number of Mel bands")
    parser.add_argument("--segment_duration", type=float, default=5.0, 
                        help="Duration of audio segments in seconds")
    parser.add_argument("--overlap", type=float, default=0.5, 
                        help="Overlap between segments")
    
    # Model options
    parser.add_argument("--input_channels", type=int, default=1, 
                        help="Number of input channels")
    parser.add_argument("--hidden_channels", type=str, default="32,64,128,256", 
                        help="Hidden channels for CNN layers (comma-separated)")
    parser.add_argument("--lstm_hidden_size", type=int, default=256, 
                        help="Hidden size for LSTM layers")
    parser.add_argument("--attention_dim", type=int, default=128, 
                        help="Dimension for attention mechanism")
    parser.add_argument("--num_pitches", type=int, default=128, 
                        help="Number of possible pitches")
    parser.add_argument("--max_duration_frames", type=int, default=100, 
                        help="Maximum duration in frames")
    parser.add_argument("--num_instruments", type=int, default=10, 
                        help="Number of possible instruments")
    parser.add_argument("--dropout", type=float, default=0.2, 
                        help="Dropout rate")
    
    # Training options
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, 
                        help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, 
                        help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adam", 
                        help="Optimizer type (adam, sgd, adamw)")
    parser.add_argument("--scheduler", type=str, default="cosine", 
                        help="Scheduler type (step, cosine, plateau)")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Patience for early stopping")
    parser.add_argument("--save_every", type=int, default=5, 
                        help="Save checkpoint every n epochs")
    parser.add_argument("--resume_from", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of worker threads for DataLoader")
    
    # Output options
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", 
                        help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", 
                        help="Directory to save logs")
    parser.add_argument("--experiment_name", type=str, default=None, 
                        help="Name of experiment (default: timestamp)")
    
    args = parser.parse_args()
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    log_dir = os.path.join(args.log_dir, args.experiment_name)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Experiment: {args.experiment_name}")
    print(f"Configuration saved to {config_path}")
    
    # Download dataset if requested
    if args.download:
        print("Downloading MAESTRO dataset...")
        dataset_dir = download_maestro(args.dataset_dir)
    else:
        dataset_dir = args.dataset_dir
    
    # Preprocess dataset if requested
    if args.preprocess:
        print("Preprocessing MAESTRO dataset...")
        
        # Set maximum items per split if specified
        max_items_per_split = None
        if args.max_items is not None:
            max_items_per_split = {
                'train': args.max_items,
                'validation': args.max_items // 5,
                'test': args.max_items // 5
            }
        
        # Preprocess dataset
        stats = preprocess_maestro(
            dataset_dir=dataset_dir,
            output_dir=args.preprocessed_dir,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            segment_duration=args.segment_duration,
            overlap=args.overlap,
            max_items_per_split=max_items_per_split
        )
        
        print("Preprocessing completed!")
        print(f"Total items: {stats['total_items']}")
        print(f"Total segments: {stats['total_segments']}")
        print(f"Processed items: {stats['processed_items']}")
        print(f"Skipped items: {stats['skipped_items']}")
        
        for split, split_stats in stats['split_stats'].items():
            print(f"{split}: {split_stats['items']} items, {split_stats['segments']} segments")
    
    # Create datasets and data loaders
    print("Creating datasets and data loaders...")
    transforms = {
        'train': create_train_transform(),
        'validation': create_validation_transform(),
        'test': create_test_transform()
    }
    
    dataloaders = create_datasets(
        preprocessed_dir=args.preprocessed_dir,
        batch_size=args.batch_size,
        max_duration_frames=args.max_duration_frames,
        num_pitches=args.num_pitches,
        num_workers=args.num_workers,
        transforms=transforms
    )
    
    # Parse hidden channels
    hidden_channels = [int(c) for c in args.hidden_channels.split(',')]
    
    # Create model
    print("Creating model...")
    model = MusicTranscriptionModel(
        input_channels=args.input_channels,
        hidden_channels=hidden_channels,
        lstm_hidden_size=args.lstm_hidden_size,
        attention_dim=args.attention_dim,
        num_pitches=args.num_pitches,
        max_duration_frames=args.max_duration_frames,
        num_instruments=args.num_instruments,
        dropout=args.dropout
    )
    
    # Train model
    print("Starting training...")
    results = train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['validation'],
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        patience=args.patience,
        save_every=args.save_every,
        resume_from=args.resume_from
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
