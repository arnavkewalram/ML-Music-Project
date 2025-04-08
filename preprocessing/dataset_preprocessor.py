"""
Dataset Preprocessing Module for Music Transcription System

This module provides functionality for preprocessing the MAESTRO dataset
for training the music transcription model. It includes functions for
downloading the dataset, extracting features, and creating PyTorch datasets.

Functions:
    download_maestro: Download the MAESTRO dataset
    preprocess_maestro: Preprocess the MAESTRO dataset
    create_datasets: Create PyTorch datasets from preprocessed data
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import librosa
import pretty_midi
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
import urllib.request
import tarfile
import shutil

from preprocessing.audio_processor import AudioLoader, FeatureExtractor
from preprocessing.midi_processor import MIDILoader, NoteEventExtractor, MIDIConverter
from preprocessing.data_loader import MAESTRODataset, create_maestro_dataloaders


def download_maestro(download_dir: str, version: str = "v3.0.0") -> str:
    """
    Download the MAESTRO dataset.
    
    Args:
        download_dir (str): Directory to download the dataset
        version (str): Version of the MAESTRO dataset
        
    Returns:
        str: Path to the extracted dataset
    """
    # Create download directory
    os.makedirs(download_dir, exist_ok=True)
    
    # URL for MAESTRO dataset
    url = f"https://storage.googleapis.com/magentadata/datasets/maestro/{version}/maestro-{version}.tar.gz"
    
    # Download path
    download_path = os.path.join(download_dir, f"maestro-{version}.tar.gz")
    
    # Extract path
    extract_path = os.path.join(download_dir, f"maestro-{version}")
    
    # Download if not already downloaded
    if not os.path.exists(download_path):
        print(f"Downloading MAESTRO dataset from {url}...")
        urllib.request.urlretrieve(url, download_path)
        print(f"Download completed: {download_path}")
    else:
        print(f"Found existing download at {download_path}")
    
    # Extract if not already extracted
    if not os.path.exists(extract_path):
        print(f"Extracting MAESTRO dataset to {extract_path}...")
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=download_dir)
        print(f"Extraction completed: {extract_path}")
    else:
        print(f"Found existing extracted dataset at {extract_path}")
    
    return extract_path


def preprocess_maestro(
    dataset_dir: str,
    output_dir: str,
    sample_rate: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    segment_duration: float = 5.0,
    overlap: float = 0.5,
    features: List[str] = None,
    splits: List[str] = None,
    max_items_per_split: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Preprocess the MAESTRO dataset for training.
    
    Args:
        dataset_dir (str): Directory containing the MAESTRO dataset
        output_dir (str): Directory to save preprocessed data
        sample_rate (int): Target sample rate for audio processing
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
        n_mels (int): Number of Mel bands
        segment_duration (float): Duration of audio segments in seconds
        overlap (float): Overlap between segments (0.0-1.0)
        features (List[str], optional): List of audio features to extract
        splits (List[str], optional): List of dataset splits to process
        max_items_per_split (Dict[str, int], optional): Maximum number of items to process per split
        
    Returns:
        Dict[str, Any]: Dictionary containing preprocessing statistics
    """
    # Default features if not provided
    if features is None:
        features = ['melspectrogram', 'mfcc', 'chroma', 'onset_strength']
    
    # Default splits if not provided
    if splits is None:
        splits = ['train', 'validation', 'test']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for each split
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Load dataset metadata
    metadata_path = os.path.join(dataset_dir, 'maestro-v3.0.0.json')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Convert to DataFrame
    metadata_df = pd.DataFrame(metadata)
    
    # Create instances for processing
    audio_loader = AudioLoader(sample_rate=sample_rate)
    feature_extractor = FeatureExtractor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    midi_loader = MIDILoader()
    note_extractor = NoteEventExtractor()
    midi_converter = MIDIConverter(
        sample_rate=sample_rate,
        hop_length=hop_length
    )
    
    # Statistics
    stats = {
        'total_items': 0,
        'total_segments': 0,
        'processed_items': 0,
        'skipped_items': 0,
        'split_stats': {split: {'items': 0, 'segments': 0} for split in splits}
    }
    
    # Process each split
    for split in splits:
        print(f"Processing {split} split...")
        
        # Filter metadata by split
        split_data = metadata_df[metadata_df['split'] == split]
        
        # Limit number of items if specified
        if max_items_per_split and split in max_items_per_split:
            split_data = split_data.head(max_items_per_split[split])
        
        stats['split_stats'][split]['items'] = len(split_data)
        stats['total_items'] += len(split_data)
        
        # Process each item
        for idx, (_, row) in enumerate(tqdm(split_data.iterrows(), total=len(split_data))):
            try:
                # Paths
                audio_path = os.path.join(dataset_dir, row['audio_filename'])
                midi_path = os.path.join(dataset_dir, row['midi_filename'])
                
                # Output path for this item
                item_output_dir = os.path.join(output_dir, split, f"item_{idx:04d}")
                os.makedirs(item_output_dir, exist_ok=True)
                
                # Load audio
                audio_data, _ = audio_loader.load_file(audio_path)
                
                # Calculate segment size in samples
                segment_samples = int(segment_duration * sample_rate)
                hop_samples = int(segment_samples * (1 - overlap))
                
                # Calculate number of segments
                num_segments = max(1, 1 + (len(audio_data) - segment_samples) // hop_samples)
                
                stats['split_stats'][split]['segments'] += num_segments
                stats['total_segments'] += num_segments
                
                # Load MIDI
                midi_data = midi_loader.load_file_pretty_midi(midi_path)
                
                # Extract note events
                notes = note_extractor.extract_notes_from_pretty_midi(midi_data)
                
                # Save metadata
                metadata_output = {
                    'canonical_composer': row['canonical_composer'],
                    'canonical_title': row['canonical_title'],
                    'year': row['year'],
                    'duration': row['duration'],
                    'audio_path': audio_path,
                    'midi_path': midi_path,
                    'num_segments': num_segments,
                    'segment_duration': segment_duration,
                    'overlap': overlap,
                    'sample_rate': sample_rate,
                    'n_fft': n_fft,
                    'hop_length': hop_length,
                    'n_mels': n_mels
                }
                
                with open(os.path.join(item_output_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata_output, f, indent=2)
                
                # Process each segment
                for segment_idx in range(num_segments):
                    # Segment start and end in samples
                    start_sample = segment_idx * hop_samples
                    end_sample = start_sample + segment_samples
                    
                    # Handle last segment
                    if end_sample > len(audio_data):
                        # Pad with zeros
                        segment = np.zeros(segment_samples)
                        segment[:len(audio_data) - start_sample] = audio_data[start_sample:]
                    else:
                        segment = audio_data[start_sample:end_sample]
                    
                    # Segment start and end in seconds
                    start_time = start_sample / sample_rate
                    end_time = end_sample / sample_rate
                    
                    # Extract features
                    segment_features = {}
                    for feature_name in features:
                        if hasattr(feature_extractor, feature_name):
                            segment_features[feature_name] = getattr(
                                feature_extractor, 
                                feature_name
                            )(segment)
                    
                    # Filter notes for this segment
                    segment_notes = [
                        note for note in notes
                        if (note['onset_time'] >= start_time and note['onset_time'] < end_time) or
                           (note['offset_time'] > start_time and note['offset_time'] <= end_time) or
                           (note['onset_time'] <= start_time and note['offset_time'] >= end_time)
                    ]
                    
                    # Adjust note times relative to segment start
                    adjusted_notes = []
                    for note in segment_notes:
                        adjusted_note = note.copy()
                        adjusted_note['onset_time'] = max(0, note['onset_time'] - start_time)
                        adjusted_note['offset_time'] = min(segment_duration, note['offset_time'] - start_time)
                        adjusted_note['duration_time'] = adjusted_note['offset_time'] - adjusted_note['onset_time']
                        adjusted_notes.append(adjusted_note)
                    
                    # Create targets
                    # 1. Onsets
                    onset_times = np.array([note['onset_time'] for note in adjusted_notes])
                    onset_frames = midi_converter.time_to_frames(onset_times)
                    
                    # 2. Pitches
                    pitches = np.array([note['pitch'] for note in adjusted_notes])
                    
                    # 3. Durations
                    durations = np.array([note['duration_time'] for note in adjusted_notes])
                    duration_frames = np.round(durations * sample_rate / hop_length).astype(int)
                    
                    # 4. Velocities
                    velocities = np.array([note['velocity'] for note in adjusted_notes])
                    
                    # Save segment data
                    segment_output_path = os.path.join(item_output_dir, f"segment_{segment_idx:04d}.npz")
                    np.savez_compressed(
                        segment_output_path,
                        features=segment_features,
                        notes=adjusted_notes,
                        onset_times=onset_times,
                        onset_frames=onset_frames,
                        pitches=pitches,
                        durations=durations,
                        duration_frames=duration_frames,
                        velocities=velocities,
                        start_time=start_time,
                        end_time=end_time
                    )
                
                stats['processed_items'] += 1
                
            except Exception as e:
                print(f"Error processing {row['audio_filename']}: {e}")
                stats['skipped_items'] += 1
    
    # Save statistics
    with open(os.path.join(output_dir, 'preprocessing_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


class PreprocessedMAESTRODataset(Dataset):
    """
    PyTorch dataset for preprocessed MAESTRO data.
    
    Attributes:
        data_dir (str): Directory containing preprocessed data
        split (str): Dataset split ('train', 'validation', 'test')
        segment_files (List[str]): List of segment files
        max_duration_frames (int): Maximum duration in frames
        num_pitches (int): Number of possible pitches
        transform (callable, optional): Optional transform to be applied on a sample
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_duration_frames: int = 100,
        num_pitches: int = 128,
        transform: Optional[callable] = None
    ):
        """
        Initialize the PreprocessedMAESTRODataset with specified parameters.
        
        Args:
            data_dir (str): Directory containing preprocessed data
            split (str): Dataset split ('train', 'validation', 'test')
            max_duration_frames (int): Maximum duration in frames
            num_pitches (int): Number of possible pitches
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.split = split
        self.max_duration_frames = max_duration_frames
        self.num_pitches = num_pitches
        self.transform = transform
        
        # Find all segment files
        self.segment_files = []
        split_dir = os.path.join(data_dir, split)
        
        for item_dir in os.listdir(split_dir):
            item_path = os.path.join(split_dir, item_dir)
            if os.path.isdir(item_path):
                for file in os.listdir(item_path):
                    if file.startswith('segment_') and file.endswith('.npz'):
                        self.segment_files.append(os.path.join(item_path, file))
        
        # Sort segment files for reproducibility
        self.segment_files.sort()
    
    def __len__(self) -> int:
        """Return the number of segments in the dataset."""
        return len(self.segment_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing features and targets
        """
        # Load segment data
        segment_path = self.segment_files[idx]
        data = np.load(segment_path, allow_pickle=True)
        
        # Extract features
        features = data['features'].item()
        
        # Convert features to tensors
        feature_tensors = {}
        for name, feature in features.items():
            feature_tensors[name] = torch.from_numpy(feature).float()
        
        # Extract targets
        onset_frames = data['onset_frames']
        pitches = data['pitches']
        duration_frames = data['duration_frames']
        velocities = data['velocities']
        
        # Create target tensors
        # 1. Onsets - binary tensor indicating onset frames
        num_frames = feature_tensors['melspectrogram'].shape[1]
        onset_target = torch.zeros(num_frames)
        for frame in onset_frames:
            if 0 <= frame < num_frames:
                onset_target[frame] = 1.0
        
        # 2. Pitches - multi-hot tensor for active pitches at each frame
        pitch_target = torch.zeros(num_frames, self.num_pitches)
        for i, frame in enumerate(onset_frames):
            if 0 <= frame < num_frames and 0 <= pitches[i] < self.num_pitches:
                # Set pitch active at onset frame
                pitch_target[frame, pitches[i]] = 1.0
                
                # Also set pitch active for duration frames if within bounds
                end_frame = min(num_frames, frame + duration_frames[i])
                for j in range(frame + 1, end_frame):
                    pitch_target[j, pitches[i]] = 1.0
        
        # 3. Durations - for each onset frame, the duration in frames
        duration_target = torch.zeros(num_frames, dtype=torch.long)
        for i, frame in enumerate(onset_frames):
            if 0 <= frame < num_frames:
                # Clip duration to max_duration_frames
                duration_target[frame] = min(duration_frames[i], self.max_duration_frames - 1)
        
        # 4. Velocities - for each onset frame, the velocity
        velocity_target = torch.zeros(num_frames)
        for i, frame in enumerate(onset_frames):
            if 0 <= frame < num_frames:
                velocity_target[frame] = velocities[i] / 127.0  # Normalize to [0, 1]
        
        # Create sample
        sample = {
            'melspectrogram': feature_tensors['melspectrogram'].unsqueeze(0),  # Add channel dimension
            'onset_target': onset_target,
            'pitch_target': pitch_target,
            'duration_target': duration_target,
            'velocity_target': velocity_target
        }
        
        # Add other features if available
        for name, tensor in feature_tensors.items():
            if name != 'melspectrogram':
                sample[name] = tensor.unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_datasets(
    preprocessed_dir: str,
    batch_size: int = 32,
    max_duration_frames: int = 100,
    num_pitches: int = 128,
    num_workers: int = 4,
    transforms: Optional[Dict[str, callable]] = None
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders from preprocessed data.
    
    Args:
        preprocessed_dir (str): Directory containing preprocessed data
        batch_size (int): Batch size for DataLoader
        max_duration_frames (int): Maximum duration in frames
        num_pitches (int): Number of possible pitches
        num_workers (int): Number of worker threads for DataLoader
        transforms (Dict[str, callable], optional): Dictionary of transforms for each split
        
    Returns:
        Dict[str, DataLoader]: Dictionary of DataLoaders for each split
    """
    # Default transforms if not provided
    if transforms is None:
        transforms = {
            'train': None,
            'validation': None,
            'test': None
        }
    
    # Create datasets for each split
    train_dataset = PreprocessedMAESTRODataset(
        data_dir=preprocessed_dir,
        split='train',
        max_duration_frames=max_duration_frames,
        num_pitches=num_pitches,
        transform=transforms.get('train')
    )
    
    validation_dataset = PreprocessedMAESTRODataset(
        data_dir=preprocessed_dir,
        split='validation',
        max_duration_frames=max_duration_frames,
        num_pitches=num_pitches,
        transform=transforms.get('validation')
    )
    
    test_dataset = PreprocessedMAESTRODataset(
        data_dir=preprocessed_dir,
        split='test',
        max_duration_frames=max_duration_frames,
        num_pitches=num_pitches,
        transform=transforms.get('test')
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'validation': validation_loader,
        'test': test_loader
    }


class DataAugmentation:
    """
    Class for data augmentation techniques for music transcription.
    
    Methods:
        pitch_shift: Shift pitch of audio and corresponding MIDI
        time_stretch: Stretch audio and corresponding MIDI in time
        add_noise: Add random noise to audio
        random_eq: Apply random equalization to audio
    """
    
    @staticmethod
    def pitch_shift(sample: Dict[str, torch.Tensor], n_steps: int) -> Dict[str, torch.Tensor]:
        """
        Shift pitch of audio features and corresponding MIDI targets.
        
        Args:
            sample (Dict[str, torch.Tensor]): Dictionary containing features and targets
            n_steps (int): Number of semitones to shift
            
        Returns:
            Dict[str, torch.Tensor]: Augmented sample
        """
        # Create a copy of the sample
        augmented = {k: v.clone() for k, v in sample.items()}
        
        # Shift pitch targets
        if 'pitch_target' in augmented:
            pitch_target = augmented['pitch_target']
            shifted_pitch_target = torch.zeros_like(pitch_target)
            
            # Shift each pitch by n_steps
            for i in range(pitch_target.shape[1]):
                shifted_idx = i + n_steps
                if 0 <= shifted_idx < pitch_target.shape[1]:
                    shifted_pitch_target[:, shifted_idx] = pitch_target[:, i]
            
            augmented['pitch_target'] = shifted_pitch_target
        
        # Note: For real pitch shifting of audio features, we would need to apply
        # more complex transformations to the spectrograms. This is a simplified version.
        
        return augmented
    
    @staticmethod
    def time_stretch(sample: Dict[str, torch.Tensor], rate: float) -> Dict[str, torch.Tensor]:
        """
        Stretch audio features and corresponding MIDI targets in time.
        
        Args:
            sample (Dict[str, torch.Tensor]): Dictionary containing features and targets
            rate (float): Stretch factor (>1 for slower, <1 for faster)
            
        Returns:
            Dict[str, torch.Tensor]: Augmented sample
        """
        # Create a copy of the sample
        augmented = {}
        
        # Time-stretch features
        for key, value in sample.items():
            if key in ['melspectrogram', 'mfcc', 'chroma', 'onset_strength']:
                # For 2D features (feature_dim, time)
                if value.dim() == 3:  # (channels, feature_dim, time)
                    orig_shape = value.shape
                    channels, feature_dim, time_dim = orig_shape
                    
                    # Interpolate along time dimension
                    new_time_dim = int(time_dim / rate)
                    if new_time_dim <= 0:
                        new_time_dim = 1
                    
                    # Reshape for interpolation
                    reshaped = value.view(channels * feature_dim, 1, time_dim)
                    
                    # Apply interpolation
                    stretched = torch.nn.functional.interpolate(
                        reshaped,
                        size=new_time_dim,
                        mode='linear',
                        align_corners=False
                    )
                    
                    # Reshape back
                    stretched = stretched.view(channels, feature_dim, new_time_dim)
                    
                    augmented[key] = stretched
                else:
                    # Skip non-feature tensors
                    augmented[key] = value
            elif key in ['onset_target', 'velocity_target']:
                # For 1D targets (time)
                time_dim = value.shape[0]
                new_time_dim = int(time_dim / rate)
                if new_time_dim <= 0:
                    new_time_dim = 1
                
                # Apply interpolation
                stretched = torch.nn.functional.interpolate(
                    value.view(1, 1, -1),
                    size=new_time_dim,
                    mode='linear',
                    align_corners=False
                )
                
                augmented[key] = stretched.view(-1)
            elif key == 'pitch_target':
                # For 2D targets (time, pitch)
                time_dim, pitch_dim = value.shape
                new_time_dim = int(time_dim / rate)
                if new_time_dim <= 0:
                    new_time_dim = 1
                
                # Apply interpolation
                stretched = torch.nn.functional.interpolate(
                    value.view(1, pitch_dim, time_dim),
                    size=new_time_dim,
                    mode='linear',
                    align_corners=False
                )
                
                augmented[key] = stretched.view(new_time_dim, pitch_dim)
            elif key == 'duration_target':
                # For duration targets, we need to adjust the durations
                time_dim = value.shape[0]
                new_time_dim = int(time_dim / rate)
                if new_time_dim <= 0:
                    new_time_dim = 1
                
                # Create new duration target
                new_duration = torch.zeros(new_time_dim, dtype=value.dtype)
                
                # Scale durations
                for i in range(time_dim):
                    if value[i] > 0:
                        new_idx = int(i / rate)
                        if new_idx < new_time_dim:
                            new_duration[new_idx] = max(1, int(value[i] / rate))
                
                augmented[key] = new_duration
            else:
                # Pass through other tensors
                augmented[key] = value
        
        return augmented
    
    @staticmethod
    def add_noise(sample: Dict[str, torch.Tensor], noise_level: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Add random noise to audio features.
        
        Args:
            sample (Dict[str, torch.Tensor]): Dictionary containing features and targets
            noise_level (float): Level of noise to add
            
        Returns:
            Dict[str, torch.Tensor]: Augmented sample
        """
        # Create a copy of the sample
        augmented = {k: v.clone() for k, v in sample.items()}
        
        # Add noise to features
        for key in ['melspectrogram', 'mfcc', 'chroma', 'onset_strength']:
            if key in augmented:
                feature = augmented[key]
                noise = torch.randn_like(feature) * noise_level * feature.mean()
                augmented[key] = feature + noise
        
        return augmented
    
    @staticmethod
    def random_eq(sample: Dict[str, torch.Tensor], max_gain: float = 6.0) -> Dict[str, torch.Tensor]:
        """
        Apply random equalization to audio features.
        
        Args:
            sample (Dict[str, torch.Tensor]): Dictionary containing features and targets
            max_gain (float): Maximum gain in dB
            
        Returns:
            Dict[str, torch.Tensor]: Augmented sample
        """
        # Create a copy of the sample
        augmented = {k: v.clone() for k, v in sample.items()}
        
        # Apply EQ to mel spectrogram
        if 'melspectrogram' in augmented:
            mel_spec = augmented['melspectrogram']
            
            # Create random gains for each frequency band
            n_mels = mel_spec.shape[1]
            gains = torch.rand(n_mels) * max_gain - (max_gain / 2)  # -max_gain/2 to +max_gain/2
            
            # Convert from dB to linear
            gain_factors = 10 ** (gains / 20)
            
            # Apply gains
            gain_factors = gain_factors.view(1, -1, 1)  # Shape for broadcasting
            augmented['melspectrogram'] = mel_spec * gain_factors
        
        return augmented
    
    @staticmethod
    def combine_augmentations(
        sample: Dict[str, torch.Tensor],
        p_pitch_shift: float = 0.3,
        p_time_stretch: float = 0.3,
        p_add_noise: float = 0.3,
        p_random_eq: float = 0.3
    ) -> Dict[str, torch.Tensor]:
        """
        Apply multiple augmentations with given probabilities.
        
        Args:
            sample (Dict[str, torch.Tensor]): Dictionary containing features and targets
            p_pitch_shift (float): Probability of applying pitch shift
            p_time_stretch (float): Probability of applying time stretch
            p_add_noise (float): Probability of adding noise
            p_random_eq (float): Probability of applying random EQ
            
        Returns:
            Dict[str, torch.Tensor]: Augmented sample
        """
        augmented = {k: v.clone() for k, v in sample.items()}
        
        # Apply pitch shift
        if torch.rand(1).item() < p_pitch_shift:
            n_steps = torch.randint(-6, 7, (1,)).item()  # -6 to +6 semitones
            augmented = DataAugmentation.pitch_shift(augmented, n_steps)
        
        # Apply time stretch
        if torch.rand(1).item() < p_time_stretch:
            rate = 0.8 + 0.4 * torch.rand(1).item()  # 0.8 to 1.2
            augmented = DataAugmentation.time_stretch(augmented, rate)
        
        # Add noise
        if torch.rand(1).item() < p_add_noise:
            noise_level = 0.001 + 0.019 * torch.rand(1).item()  # 0.001 to 0.02
            augmented = DataAugmentation.add_noise(augmented, noise_level)
        
        # Apply random EQ
        if torch.rand(1).item() < p_random_eq:
            max_gain = 3.0 + 3.0 * torch.rand(1).item()  # 3.0 to 6.0 dB
            augmented = DataAugmentation.random_eq(augmented, max_gain)
        
        return augmented


def create_train_transform() -> callable:
    """
    Create transform for training data.
    
    Returns:
        callable: Transform function
    """
    def transform(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return DataAugmentation.combine_augmentations(sample)
    
    return transform


def create_validation_transform() -> callable:
    """
    Create transform for validation data.
    
    Returns:
        callable: Transform function
    """
    def transform(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return sample
    
    return transform


def create_test_transform() -> callable:
    """
    Create transform for test data.
    
    Returns:
        callable: Transform function
    """
    def transform(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return sample
    
    return transform


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess MAESTRO dataset")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to MAESTRO dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for preprocessed data")
    parser.add_argument("--download", action="store_true", help="Download MAESTRO dataset")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate for audio processing")
    parser.add_argument("--n_fft", type=int, default=2048, help="FFT window size")
    parser.add_argument("--hop_length", type=int, default=512, help="Hop length for STFT")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of Mel bands")
    parser.add_argument("--segment_duration", type=float, default=5.0, help="Duration of audio segments in seconds")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap between segments")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items to process per split")
    
    args = parser.parse_args()
    
    # Download dataset if requested
    if args.download:
        dataset_dir = download_maestro(args.dataset_dir)
    else:
        dataset_dir = args.dataset_dir
    
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
        output_dir=args.output_dir,
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
