"""
Data Loader Module for Music Transcription System

This module provides functionality for loading and preprocessing the MAESTRO dataset.
It includes classes and functions for creating PyTorch datasets and data loaders
for audio-MIDI pairs.

Classes:
    MAESTRODataset: PyTorch dataset for the MAESTRO dataset
    AudioMIDIPair: Class representing an audio-MIDI pair
    DataPreprocessor: Handles preprocessing of audio and MIDI data
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

from preprocessing.audio_processor import AudioLoader, FeatureExtractor
from preprocessing.midi_processor import MIDILoader, NoteEventExtractor


class AudioMIDIPair:
    """
    Class representing an audio-MIDI pair.
    
    Attributes:
        audio_path (str): Path to the audio file
        midi_path (str): Path to the MIDI file
        metadata (Dict[str, Any]): Additional metadata
    """
    
    def __init__(
        self, 
        audio_path: str, 
        midi_path: str, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the AudioMIDIPair with specified paths and metadata.
        
        Args:
            audio_path (str): Path to the audio file
            midi_path (str): Path to the MIDI file
            metadata (Dict[str, Any], optional): Additional metadata
        """
        self.audio_path = audio_path
        self.midi_path = midi_path
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        """String representation of the AudioMIDIPair."""
        return f"AudioMIDIPair(audio_path={self.audio_path}, midi_path={self.midi_path})"


class DataPreprocessor:
    """
    Class for preprocessing audio and MIDI data.
    
    Attributes:
        sample_rate (int): Target sample rate for audio processing
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
        n_mels (int): Number of Mel bands
        segment_duration (float): Duration of audio segments in seconds
        overlap (float): Overlap between segments (0.0-1.0)
    """
    
    def __init__(
        self, 
        sample_rate: int = 22050, 
        n_fft: int = 2048, 
        hop_length: int = 512,
        n_mels: int = 128,
        segment_duration: float = 5.0,
        overlap: float = 0.5
    ):
        """
        Initialize the DataPreprocessor with specified parameters.
        
        Args:
            sample_rate (int): Target sample rate for audio processing
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
            n_mels (int): Number of Mel bands
            segment_duration (float): Duration of audio segments in seconds
            overlap (float): Overlap between segments (0.0-1.0)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.segment_duration = segment_duration
        self.overlap = overlap
        
        self.audio_loader = AudioLoader(sample_rate=sample_rate)
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.midi_loader = MIDILoader()
        self.note_extractor = NoteEventExtractor()
    
    def preprocess_audio(self, audio_path: str, features: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Preprocess an audio file and extract specified features.
        
        Args:
            audio_path (str): Path to the audio file
            features (List[str], optional): List of features to extract.
                                           If None, extracts all features.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of extracted features
        """
        # Load audio file
        audio_data, _ = self.audio_loader.load_file(audio_path)
        
        # Extract features
        if features is None:
            return self.feature_extractor.extract_all_features(audio_data)
        
        result = {}
        for feature in features:
            if hasattr(self.feature_extractor, feature):
                result[feature] = getattr(self.feature_extractor, feature)(audio_data)
        
        return result
    
    def preprocess_midi(self, midi_path: str) -> Dict[str, Any]:
        """
        Preprocess a MIDI file and extract note events.
        
        Args:
            midi_path (str): Path to the MIDI file
            
        Returns:
            Dict[str, Any]: Dictionary of extracted MIDI data
        """
        # Load MIDI file
        midi_data = self.midi_loader.load_file_pretty_midi(midi_path)
        
        # Extract note events
        notes = self.note_extractor.extract_notes_from_pretty_midi(midi_data)
        
        # Get piano roll
        piano_roll = self.note_extractor.get_piano_roll(midi_data)
        
        # Extract tempo and time signature
        tempo = midi_data.get_tempo_changes()
        
        return {
            'notes': notes,
            'piano_roll': piano_roll,
            'tempo': tempo,
            'midi_data': midi_data
        }
    
    def segment_audio(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """
        Segment audio data into overlapping chunks.
        
        Args:
            audio_data (np.ndarray): Audio time series
            
        Returns:
            List[np.ndarray]: List of audio segments
        """
        segment_samples = int(self.segment_duration * self.sample_rate)
        hop_samples = int(segment_samples * (1 - self.overlap))
        
        # Calculate number of segments
        num_segments = max(1, 1 + (len(audio_data) - segment_samples) // hop_samples)
        
        segments = []
        for i in range(num_segments):
            start = i * hop_samples
            end = start + segment_samples
            
            # Handle last segment
            if end > len(audio_data):
                # Pad with zeros
                segment = np.zeros(segment_samples)
                segment[:len(audio_data) - start] = audio_data[start:]
            else:
                segment = audio_data[start:end]
            
            segments.append(segment)
        
        return segments
    
    def segment_features(self, features: Dict[str, np.ndarray], segment_frames: int, hop_frames: int) -> List[Dict[str, np.ndarray]]:
        """
        Segment feature matrices into overlapping chunks.
        
        Args:
            features (Dict[str, np.ndarray]): Dictionary of feature matrices
            segment_frames (int): Number of frames per segment
            hop_frames (int): Number of frames between segment starts
            
        Returns:
            List[Dict[str, np.ndarray]]: List of segmented feature dictionaries
        """
        # Get the number of frames in the time dimension
        # Assuming all features have the same number of frames in the time dimension
        feature_name = list(features.keys())[0]
        feature_matrix = features[feature_name]
        
        # Determine the time dimension (usually the second dimension, index 1)
        time_dim = 1 if feature_matrix.ndim > 1 else 0
        num_frames = feature_matrix.shape[time_dim]
        
        # Calculate number of segments
        num_segments = max(1, 1 + (num_frames - segment_frames) // hop_frames)
        
        segmented_features = []
        for i in range(num_segments):
            start = i * hop_frames
            end = start + segment_frames
            
            segment = {}
            for name, matrix in features.items():
                # Handle different feature shapes
                if matrix.ndim == 1:  # 1D feature (e.g., onset strength)
                    if end > len(matrix):
                        # Pad with zeros
                        padded = np.zeros(segment_frames)
                        padded[:len(matrix) - start] = matrix[start:]
                        segment[name] = padded
                    else:
                        segment[name] = matrix[start:end]
                else:  # 2D feature (e.g., spectrogram)
                    if time_dim == 1:  # (freq, time)
                        if end > matrix.shape[1]:
                            # Pad with zeros
                            padded = np.zeros((matrix.shape[0], segment_frames))
                            padded[:, :matrix.shape[1] - start] = matrix[:, start:]
                            segment[name] = padded
                        else:
                            segment[name] = matrix[:, start:end]
                    else:  # (time, freq)
                        if end > matrix.shape[0]:
                            # Pad with zeros
                            padded = np.zeros((segment_frames, matrix.shape[1]))
                            padded[:matrix.shape[0] - start, :] = matrix[start:, :]
                            segment[name] = padded
                        else:
                            segment[name] = matrix[start:end, :]
            
            segmented_features.append(segment)
        
        return segmented_features
    
    def align_midi_to_audio(self, midi_data: Dict[str, Any], audio_segments: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Align MIDI data to audio segments.
        
        Args:
            midi_data (Dict[str, Any]): Dictionary of MIDI data
            audio_segments (List[np.ndarray]): List of audio segments
            
        Returns:
            List[Dict[str, Any]]: List of aligned MIDI data for each segment
        """
        segment_duration = self.segment_duration
        hop_duration = segment_duration * (1 - self.overlap)
        
        notes = midi_data['notes']
        aligned_midi = []
        
        for i in range(len(audio_segments)):
            start_time = i * hop_duration
            end_time = start_time + segment_duration
            
            # Filter notes that fall within this segment
            segment_notes = [
                note for note in notes
                if (note['onset_time'] >= start_time and note['onset_time'] < end_time) or
                   (note['offset_time'] > start_time and note['offset_time'] <= end_time) or
                   (note['onset_time'] <= start_time and note['offset_time'] >= end_time)
            ]
            
            # Adjust note times relative to segment start
            for note in segment_notes:
                note = note.copy()  # Create a copy to avoid modifying the original
                note['onset_time'] = max(0, note['onset_time'] - start_time)
                note['offset_time'] = min(segment_duration, note['offset_time'] - start_time)
                note['duration_time'] = note['offset_time'] - note['onset_time']
            
            aligned_midi.append({
                'notes': segment_notes,
                'segment_start': start_time,
                'segment_end': end_time
            })
        
        return aligned_midi


class MAESTRODataset(Dataset):
    """
    PyTorch dataset for the MAESTRO dataset.
    
    Attributes:
        root_dir (str): Root directory of the MAESTRO dataset
        split (str): Dataset split ('train', 'validation', 'test')
        preprocessor (DataPreprocessor): Data preprocessor instance
        pairs (List[AudioMIDIPair]): List of audio-MIDI pairs
        features (List[str]): List of audio features to extract
        transform (callable, optional): Optional transform to be applied on a sample
    """
    
    def __init__(
        self, 
        root_dir: str,
        split: str = 'train',
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        segment_duration: float = 5.0,
        overlap: float = 0.5,
        features: List[str] = None,
        transform: Optional[callable] = None,
        precompute: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the MAESTRODataset with specified parameters.
        
        Args:
            root_dir (str): Root directory of the MAESTRO dataset
            split (str): Dataset split ('train', 'validation', 'test')
            sample_rate (int): Target sample rate for audio processing
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
            n_mels (int): Number of Mel bands
            segment_duration (float): Duration of audio segments in seconds
            overlap (float): Overlap between segments (0.0-1.0)
            features (List[str], optional): List of audio features to extract.
                                           If None, extracts all features.
            transform (callable, optional): Optional transform to be applied on a sample
            precompute (bool): Whether to precompute and cache features
            cache_dir (str, optional): Directory to store cached features
        """
        self.root_dir = root_dir
        self.split = split
        self.features = features or ['melspectrogram', 'mfcc', 'chroma', 'onset_strength']
        self.transform = transform
        self.precompute = precompute
        self.cache_dir = cache_dir
        
        # Create preprocessor
        self.preprocessor = DataPreprocessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            segment_duration=segment_duration,
            overlap=overlap
        )
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        
        # Filter by split
        split_data = self.metadata[self.metadata['split'] == split]
        
        # Create audio-MIDI pairs
        self.pairs = []
        for _, row in split_data.iterrows():
            audio_path = os.path.join(root_dir, row['audio_filename'])
            midi_path = os.path.join(root_dir, row['midi_filename'])
            
            # Create metadata dictionary
            metadata = {
                'canonical_composer': row['canonical_composer'],
                'canonical_title': row['canonical_title'],
                'year': row['year'],
                'duration': row['duration']
            }
            
            self.pairs.append(AudioMIDIPair(audio_path, midi_path, metadata))
        
        # Create cache directory if needed
        if self.precompute and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Precompute features if requested
        if self.precompute:
            self._precompute_features()
    
    def _load_metadata(self) -> pd.DataFrame:
        """
        Load the MAESTRO dataset metadata.
        
        Returns:
            pd.DataFrame: DataFrame containing metadata
        """
        metadata_path = os.path.join(self.root_dir, 'maestro-v3.0.0.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return pd.DataFrame(metadata)
    
    def _precompute_features(self) -> None:
        """Precompute and cache features for all audio files."""
        import tqdm
        
        print(f"Precomputing features for {len(self.pairs)} audio files...")
        for i, pair in enumerate(tqdm.tqdm(self.pairs)):
            # Generate cache path
            cache_path = os.path.join(
                self.cache_dir, 
                f"{self.split}_{i}_features.npz"
            )
            
            # Skip if already cached
            if os.path.exists(cache_path):
                continue
            
            # Load and preprocess audio
            audio_data, _ = self.preprocessor.audio_loader.load_file(pair.audio_path)
            
            # Segment audio
            audio_segments = self.preprocessor.segment_audio(audio_data)
            
            # Extract features for each segment
            all_features = []
            for segment in audio_segments:
                features = {}
                for feature_name in self.features:
                    if hasattr(self.preprocessor.feature_extractor, feature_name):
                        features[feature_name] = getattr(
                            self.preprocessor.feature_extractor, 
                            feature_name
                        )(segment)
                all_features.append(features)
            
            # Preprocess MIDI
            midi_data = self.preprocessor.preprocess_midi(pair.midi_path)
            
            # Align MIDI to audio segments
            aligned_midi = self.preprocessor.align_midi_to_audio(midi_data, audio_segments)
            
            # Save to cache
            np.savez_compressed(
                cache_path,
                features=all_features,
                midi=aligned_midi,
                metadata=pair.metadata
            )
    
    def __len__(self) -> int:
        """Return the number of audio-MIDI pairs in the dataset."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Dict[str, Any]: Dictionary containing audio features and MIDI data
        """
        pair = self.pairs[idx]
        
        # Check if features are cached
        if self.precompute and self.cache_dir:
            cache_path = os.path.join(
                self.cache_dir, 
                f"{self.split}_{idx}_features.npz"
            )
            
            if os.path.exists(cache_path):
                # Load from cache
                cached_data = np.load(cache_path, allow_pickle=True)
                features = cached_data['features'].item()
                midi_data = cached_data['midi'].item()
                metadata = cached_data['metadata'].item()
                
                sample = {
                    'features': features,
                    'midi': midi_data,
                    'metadata': metadata
                }
                
                if self.transform:
                    sample = self.transform(sample)
                
                return sample
        
        # Load and preprocess audio
        audio_data, _ = self.preprocessor.audio_loader.load_file(pair.audio_path)
        
        # Segment audio
        audio_segments = self.preprocessor.segment_audio(audio_data)
        
        # Extract features for each segment
        all_features = []
        for segment in audio_segments:
            features = {}
            for feature_name in self.features:
                if hasattr(self.preprocessor.feature_extractor, feature_name):
                    features[feature_name] = getattr(
                        self.preprocessor.feature_extractor, 
                        feature_name
                    )(segment)
            all_features.append(features)
        
        # Preprocess MIDI
        midi_data = self.preprocessor.preprocess_midi(pair.midi_path)
        
        # Align MIDI to audio segments
        aligned_midi = self.preprocessor.align_midi_to_audio(midi_data, audio_segments)
        
        # Create sample
        sample = {
            'features': all_features,
            'midi': aligned_midi,
            'metadata': pair.metadata
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_maestro_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    sample_rate: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    segment_duration: float = 5.0,
    overlap: float = 0.5,
    features: List[str] = None,
    transform: Optional[callable] = None,
    precompute: bool = False,
    cache_dir: Optional[str] = None,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for the MAESTRO dataset.
    
    Args:
        root_dir (str): Root directory of the MAESTRO dataset
        batch_size (int): Batch size for DataLoader
        sample_rate (int): Target sample rate for audio processing
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
        n_mels (int): Number of Mel bands
        segment_duration (float): Duration of audio segments in seconds
        overlap (float): Overlap between segments (0.0-1.0)
        features (List[str], optional): List of audio features to extract
        transform (callable, optional): Optional transform to be applied on a sample
        precompute (bool): Whether to precompute and cache features
        cache_dir (str, optional): Directory to store cached features
        num_workers (int): Number of worker threads for DataLoader
        
    Returns:
        Dict[str, DataLoader]: Dictionary of DataLoaders for each split
    """
    # Create datasets for each split
    train_dataset = MAESTRODataset(
        root_dir=root_dir,
        split='train',
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        segment_duration=segment_duration,
        overlap=overlap,
        features=features,
        transform=transform,
        precompute=precompute,
        cache_dir=cache_dir
    )
    
    validation_dataset = MAESTRODataset(
        root_dir=root_dir,
        split='validation',
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        segment_duration=segment_duration,
        overlap=overlap,
        features=features,
        transform=transform,
        precompute=precompute,
        cache_dir=cache_dir
    )
    
    test_dataset = MAESTRODataset(
        root_dir=root_dir,
        split='test',
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        segment_duration=segment_duration,
        overlap=overlap,
        features=features,
        transform=transform,
        precompute=precompute,
        cache_dir=cache_dir
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


def download_maestro_dataset(download_dir: str) -> str:
    """
    Download the MAESTRO dataset.
    
    Args:
        download_dir (str): Directory to download the dataset
        
    Returns:
        str: Path to the downloaded dataset
    """
    import urllib.request
    import tarfile
    
    # Create download directory
    os.makedirs(download_dir, exist_ok=True)
    
    # URL for MAESTRO v3.0.0
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.tar.gz"
    
    # Download path
    download_path = os.path.join(download_dir, "maestro-v3.0.0.tar.gz")
    
    # Download if not already downloaded
    if not os.path.exists(download_path):
        print(f"Downloading MAESTRO dataset from {url}...")
        urllib.request.urlretrieve(url, download_path)
    
    # Extract if not already extracted
    extract_path = os.path.join(download_dir, "maestro-v3.0.0")
    if not os.path.exists(extract_path):
        print(f"Extracting MAESTRO dataset to {extract_path}...")
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=download_dir)
    
    return extract_path
