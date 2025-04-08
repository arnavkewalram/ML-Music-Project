"""
Audio Processor Module for Music Transcription System

This module provides functionality for processing audio files and live audio input.
It includes classes and functions for loading audio files, capturing live audio,
and extracting features using FFT/DFT sliding window approach.

Classes:
    AudioLoader: Handles loading and basic processing of audio files
    LiveAudioHandler: Manages live audio input from microphone
    FeatureExtractor: Extracts spectral features from audio data
    BatchProcessor: Processes multiple audio files in batch mode
"""

import os
import numpy as np
import librosa
import soundfile as sf
import pyaudio
import wave
import threading
import queue
import torch
import torchaudio
from typing import Dict, List, Tuple, Optional, Union, Any


class AudioLoader:
    """
    Class for loading and basic processing of audio files.
    
    Attributes:
        sample_rate (int): Target sample rate for audio processing
        mono (bool): Whether to convert audio to mono
        normalize (bool): Whether to normalize audio amplitude
    """
    
    def __init__(self, sample_rate: int = 22050, mono: bool = True, normalize: bool = True):
        """
        Initialize the AudioLoader with specified parameters.
        
        Args:
            sample_rate (int): Target sample rate for audio processing
            mono (bool): Whether to convert audio to mono
            normalize (bool): Whether to normalize audio amplitude
        """
        self.sample_rate = sample_rate
        self.mono = mono
        self.normalize = normalize
        
    def load_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and return the audio data and sample rate.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            # Use librosa for robust audio loading with resampling
            audio_data, orig_sr = librosa.load(
                file_path, 
                sr=self.sample_rate if self.sample_rate else None,
                mono=self.mono
            )
            
            # Normalize if requested
            if self.normalize:
                audio_data = librosa.util.normalize(audio_data)
                
            return audio_data, self.sample_rate or orig_sr
            
        except Exception as e:
            raise ValueError(f"Error loading audio file: {e}")
    
    def load_file_torchaudio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load an audio file using torchaudio and return as PyTorch tensor.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            Tuple[torch.Tensor, int]: Audio data as tensor and sample rate
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            # Load audio with torchaudio
            waveform, orig_sr = torchaudio.load(file_path)
            
            # Resample if needed
            if self.sample_rate and orig_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
                waveform = resampler(waveform)
                orig_sr = self.sample_rate
            
            # Convert to mono if needed
            if self.mono and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Normalize if requested
            if self.normalize:
                waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
                
            return waveform, orig_sr
            
        except Exception as e:
            raise ValueError(f"Error loading audio file with torchaudio: {e}")
    
    def save_file(self, audio_data: np.ndarray, file_path: str, sample_rate: int = None) -> None:
        """
        Save audio data to a file.
        
        Args:
            audio_data (np.ndarray): Audio data to save
            file_path (str): Path where to save the audio file
            sample_rate (int, optional): Sample rate to use. Defaults to instance sample_rate.
            
        Raises:
            ValueError: If there's an error saving the file
        """
        sr = sample_rate or self.sample_rate
        if sr is None:
            raise ValueError("Sample rate must be specified")
        
        try:
            sf.write(file_path, audio_data, sr)
        except Exception as e:
            raise ValueError(f"Error saving audio file: {e}")


class LiveAudioHandler:
    """
    Class for handling live audio input from microphone.
    
    Attributes:
        sample_rate (int): Sample rate for audio capture
        chunk_size (int): Number of frames per buffer
        channels (int): Number of audio channels (1 for mono, 2 for stereo)
        format (int): Audio format from pyaudio
        device_index (int): Index of input device to use
    """
    
    def __init__(
        self, 
        sample_rate: int = 22050, 
        chunk_size: int = 1024, 
        channels: int = 1,
        format_type: int = pyaudio.paFloat32,
        device_index: Optional[int] = None
    ):
        """
        Initialize the LiveAudioHandler with specified parameters.
        
        Args:
            sample_rate (int): Sample rate for audio capture
            chunk_size (int): Number of frames per buffer
            channels (int): Number of audio channels (1 for mono, 2 for stereo)
            format_type (int): Audio format from pyaudio
            device_index (int, optional): Index of input device to use
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format_type
        self.device_index = device_index
        
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.stop_stream()
        if hasattr(self, 'audio_interface') and self.audio_interface:
            self.audio_interface.terminate()
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """
        List available audio input devices.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries with device information
        """
        devices = []
        for i in range(self.audio_interface.get_device_count()):
            device_info = self.audio_interface.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # Input device
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': int(device_info['defaultSampleRate'])
                })
        return devices
    
    def start_stream(self) -> None:
        """
        Start the audio stream for capturing live audio.
        
        Raises:
            RuntimeError: If there's an error starting the stream
        """
        if self.stream is not None:
            return  # Stream already started
        
        try:
            self.stream = self.audio_interface.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.is_recording = True
        except Exception as e:
            raise RuntimeError(f"Error starting audio stream: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for the audio stream.
        
        Args:
            in_data: Input audio data
            frame_count: Number of frames
            time_info: Time information
            status: Status flag
            
        Returns:
            Tuple: (None, pyaudio.paContinue)
        """
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def stop_stream(self) -> None:
        """Stop the audio stream."""
        self.is_recording = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def start_recording(self, duration: Optional[float] = None) -> None:
        """
        Start recording audio in a separate thread.
        
        Args:
            duration (float, optional): Recording duration in seconds. If None, records until stop_recording is called.
        """
        if self.recording_thread is not None and self.recording_thread.is_alive():
            return  # Already recording
        
        self.start_stream()
        self.recording_thread = threading.Thread(
            target=self._recording_worker,
            args=(duration,)
        )
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def _recording_worker(self, duration: Optional[float] = None) -> None:
        """
        Worker function for recording thread.
        
        Args:
            duration (float, optional): Recording duration in seconds
        """
        if duration is not None:
            import time
            time.sleep(duration)
            self.stop_recording()
    
    def stop_recording(self) -> np.ndarray:
        """
        Stop recording and return the recorded audio data.
        
        Returns:
            np.ndarray: Recorded audio data
        """
        self.stop_stream()
        
        # Process all data in the queue
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
        
        if not audio_data:
            return np.array([])
        
        # Convert to numpy array based on format
        if self.format == pyaudio.paFloat32:
            dtype = np.float32
        elif self.format == pyaudio.paInt16:
            dtype = np.int16
        elif self.format == pyaudio.paInt32:
            dtype = np.int32
        else:
            dtype = np.float32
        
        # Concatenate all chunks
        audio_bytes = b''.join(audio_data)
        audio_np = np.frombuffer(audio_bytes, dtype=dtype)
        
        return audio_np
    
    def save_recording(self, file_path: str, audio_data: Optional[np.ndarray] = None) -> None:
        """
        Save the recorded audio to a file.
        
        Args:
            file_path (str): Path where to save the audio file
            audio_data (np.ndarray, optional): Audio data to save. If None, uses the last recording.
            
        Raises:
            ValueError: If there's no audio data to save
        """
        if audio_data is None:
            audio_data = self.stop_recording()
        
        if len(audio_data) == 0:
            raise ValueError("No audio data to save")
        
        # Determine file format from extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.wav':
            # For WAV files, use wave module
            with wave.open(file_path, 'wb') as wf:
                if self.format == pyaudio.paFloat32:
                    # Convert float32 to int16 for WAV
                    audio_data = (audio_data * 32767).astype(np.int16)
                    wf.setsampwidth(2)  # 16-bit
                elif self.format == pyaudio.paInt16:
                    wf.setsampwidth(2)  # 16-bit
                elif self.format == pyaudio.paInt32:
                    wf.setsampwidth(4)  # 32-bit
                
                wf.setnchannels(self.channels)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
        else:
            # For other formats, use soundfile
            sf.write(file_path, audio_data, self.sample_rate)


class FeatureExtractor:
    """
    Class for extracting spectral features from audio data using FFT/DFT sliding window.
    
    Attributes:
        sample_rate (int): Sample rate of the audio data
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
        n_mels (int): Number of Mel bands
        fmin (float): Lowest frequency (in Hz)
        fmax (float): Highest frequency (in Hz)
    """
    
    def __init__(
        self, 
        sample_rate: int = 22050, 
        n_fft: int = 2048, 
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 20.0,
        fmax: Optional[float] = None
    ):
        """
        Initialize the FeatureExtractor with specified parameters.
        
        Args:
            sample_rate (int): Sample rate of the audio data
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
            n_mels (int): Number of Mel bands
            fmin (float): Lowest frequency (in Hz)
            fmax (float, optional): Highest frequency (in Hz). If None, defaults to sample_rate/2
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2
        
    def stft(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform (STFT) of the audio data.
        
        Args:
            audio_data (np.ndarray): Audio time series
            
        Returns:
            np.ndarray: Complex-valued STFT matrix
        """
        return librosa.stft(
            audio_data, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
    
    def spectrogram(self, audio_data: np.ndarray, log: bool = True) -> np.ndarray:
        """
        Compute spectrogram of the audio data.
        
        Args:
            audio_data (np.ndarray): Audio time series
            log (bool): Whether to convert to log scale (dB)
            
        Returns:
            np.ndarray: Spectrogram
        """
        stft_matrix = self.stft(audio_data)
        power_spec = np.abs(stft_matrix) ** 2
        
        if log:
            return librosa.power_to_db(power_spec, ref=np.max)
        return power_spec
    
    def melspectrogram(self, audio_data: np.ndarray, log: bool = True) -> np.ndarray:
        """
        Compute mel-scaled spectrogram of the audio data.
        
        Args:
            audio_data (np.ndarray): Audio time series
            log (bool): Whether to convert to log scale (dB)
            
        Returns:
            np.ndarray: Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        if log:
            return librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec
    
    def mfcc(self, audio_data: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
        """
        Compute Mel-frequency cepstral coefficients (MFCCs) of the audio data.
        
        Args:
            audio_data (np.ndarray): Audio time series
            n_mfcc (int): Number of MFCCs to return
            
        Returns:
            np.ndarray: MFCCs
        """
        return librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
    
    def chroma(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Compute chromagram from an audio time series.
        
        Args:
            audio_data (np.ndarray): Audio time series
            
        Returns:
            np.ndarray: Chromagram
        """
        return librosa.feature.chroma_stft(
            y=audio_data,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
    
    def onset_strength(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Compute onset strength envelope from an audio time series.
        
        Args:
            audio_data (np.ndarray): Audio time series
            
        Returns:
            np.ndarray: Onset strength envelope
        """
        return librosa.onset.onset_strength(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
    
    def extract_all_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all available features from the audio data.
        
        Args:
            audio_data (np.ndarray): Audio time series
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of extracted features
        """
        return {
            'spectrogram': self.spectrogram(audio_data),
            'melspectrogram': self.melspectrogram(audio_data),
            'mfcc': self.mfcc(audio_data),
            'chroma': self.chroma(audio_data),
            'onset_strength': self.onset_strength(audio_data)
        }
    
    def extract_features_torch(self, audio_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features using PyTorch-based operations for GPU acceleration.
        
        Args:
            audio_tensor (torch.Tensor): Audio tensor [channels, samples]
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of extracted features as tensors
        """
        # Ensure audio is mono
        if audio_tensor.dim() > 1 and audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
        
        # Squeeze if needed
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze(0)
            
        # Convert to numpy for feature extraction
        audio_np = audio_tensor.cpu().numpy()
        
        # Extract features
        features_np = self.extract_all_features(audio_np)
        
        # Convert back to torch tensors
        features_torch = {}
        for key, value in features_np.items():
            features_torch[key] = torch.from_numpy(value).float()
            
        return features_torch


class BatchProcessor:
    """
    Class for processing multiple audio files in batch mode.
    
    Attributes:
        audio_loader (AudioLoader): Instance of AudioLoader for loading audio files
        feature_extractor (FeatureExtractor): Instance of FeatureExtractor for feature extraction
    """
    
    def __init__(
        self, 
        sample_rate: int = 22050, 
        n_fft: int = 2048, 
        hop_length: int = 512,
        n_mels: int = 128
    ):
        """
        Initialize the BatchProcessor with specified parameters.
        
        Args:
            sample_rate (int): Sample rate for audio processing
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
            n_mels (int): Number of Mel bands
        """
        self.audio_loader = AudioLoader(sample_rate=sample_rate)
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
    
    def process_file(self, file_path: str, features: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Process a single audio file and extract specified features.
        
        Args:
            file_path (str): Path to the audio file
            features (List[str], optional): List of features to extract. 
                                           If None, extracts all features.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of extracted features
        """
        # Load audio file
        audio_data, _ = self.audio_loader.load_file(file_path)
        
        # Extract features
        if features is None:
            return self.feature_extractor.extract_all_features(audio_data)
        
        result = {}
        for feature in features:
            if hasattr(self.feature_extractor, feature):
                result[feature] = getattr(self.feature_extractor, feature)(audio_data)
        
        return result
    
    def process_batch(
        self, 
        file_paths: List[str], 
        features: List[str] = None,
        max_workers: int = 4
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process multiple audio files in parallel and extract specified features.
        
        Args:
            file_paths (List[str]): List of paths to audio files
            features (List[str], optional): List of features to extract.
                                           If None, extracts all features.
            max_workers (int): Maximum number of worker threads
            
        Returns:
            Dict[str, Dict[str, np.ndarray]]: Dictionary of extracted features for each file
        """
        import concurrent.futures
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, file_path, features): file_path
                for file_path in file_paths
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    results[file_path] = future.result()
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return results
