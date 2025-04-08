"""
Inference Module for Music Transcription System

This module provides functionality for performing inference with the trained model.
It includes classes and functions for real-time transcription, batch processing,
and converting model predictions to musical notation.

Classes:
    MusicTranscriber: Main class for music transcription
    RealTimeTranscriber: Class for real-time transcription from microphone input
"""

import os
import torch
import numpy as np
import librosa
import pretty_midi
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from threading import Thread
from queue import Queue
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pyaudio

from model.architecture import MusicTranscriptionModel
from preprocessing.audio_processor import AudioLoader, FeatureExtractor, LiveAudioHandler
from preprocessing.midi_processor import MIDIConverter


class MusicTranscriber:
    """
    Main class for music transcription.
    
    Attributes:
        model (MusicTranscriptionModel): Trained music transcription model
        device (torch.device): Device to use for inference
        sample_rate (int): Sample rate for audio processing
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
        n_mels (int): Number of Mel bands
        audio_loader (AudioLoader): Instance of AudioLoader for loading audio files
        feature_extractor (FeatureExtractor): Instance of FeatureExtractor for feature extraction
        midi_converter (MIDIConverter): Instance of MIDIConverter for MIDI conversion
    """
    
    def __init__(
        self,
        model: MusicTranscriptionModel,
        device: torch.device = None,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        onset_threshold: float = 0.5,
        pitch_threshold: float = 0.5
    ):
        """
        Initialize the MusicTranscriber with specified parameters.
        
        Args:
            model (MusicTranscriptionModel): Trained music transcription model
            device (torch.device, optional): Device to use for inference
            sample_rate (int): Sample rate for audio processing
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
            n_mels (int): Number of Mel bands
            onset_threshold (float): Threshold for onset detection
            pitch_threshold (float): Threshold for pitch detection
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.onset_threshold = onset_threshold
        self.pitch_threshold = pitch_threshold
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Create instances for processing
        self.audio_loader = AudioLoader(sample_rate=sample_rate)
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.midi_converter = MIDIConverter(
            sample_rate=sample_rate,
            hop_length=hop_length
        )
    
    def load_model_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {checkpoint_path}")
    
    def transcribe_file(
        self,
        audio_path: str,
        segment_duration: float = 5.0,
        overlap: float = 0.5
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
        Args:
            audio_path (str): Path to audio file
            segment_duration (float): Duration of audio segments in seconds
            overlap (float): Overlap between segments (0.0-1.0)
            
        Returns:
            Dict[str, Any]: Dictionary of transcription results
        """
        # Load audio file
        audio_data, _ = self.audio_loader.load_file(audio_path)
        
        # Transcribe audio
        return self.transcribe_audio(audio_data, segment_duration, overlap)
    
    def transcribe_audio(
        self,
        audio_data: np.ndarray,
        segment_duration: float = 5.0,
        overlap: float = 0.5
    ) -> Dict[str, Any]:
        """
        Transcribe audio data.
        
        Args:
            audio_data (np.ndarray): Audio time series
            segment_duration (float): Duration of audio segments in seconds
            overlap (float): Overlap between segments (0.0-1.0)
            
        Returns:
            Dict[str, Any]: Dictionary of transcription results
        """
        # Calculate segment size in samples
        segment_samples = int(segment_duration * self.sample_rate)
        hop_samples = int(segment_samples * (1 - overlap))
        
        # Calculate number of segments
        num_segments = max(1, 1 + (len(audio_data) - segment_samples) // hop_samples)
        
        # Initialize lists for results
        all_onsets = []
        all_pitches = []
        all_durations = []
        all_velocities = []
        segment_start_times = []
        
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
            
            # Segment start time in seconds
            start_time = start_sample / self.sample_rate
            segment_start_times.append(start_time)
            
            # Extract features
            mel_spec = self.feature_extractor.melspectrogram(segment)
            
            # Convert to tensor
            mel_spec_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Perform inference
            with torch.no_grad():
                predictions = self.model.predict(mel_spec_tensor, onset_threshold=self.onset_threshold)
            
            # Extract predictions
            onsets = predictions['onsets'].cpu().numpy()[0]  # (time_steps,)
            pitch_probs = predictions['pitch_probs'].cpu().numpy()[0]  # (time_steps, num_pitches)
            durations = predictions['durations'].cpu().numpy()[0]  # (time_steps,)
            velocity = predictions['velocity'].cpu().numpy()[0]  # (time_steps,)
            
            # Store results
            all_onsets.append((onsets, start_time))
            all_pitches.append((pitch_probs, start_time))
            all_durations.append((durations, start_time))
            all_velocities.append((velocity, start_time))
        
        # Process results to create note events
        notes = self._process_predictions(all_onsets, all_pitches, all_durations, all_velocities)
        
        # Get tempo and time signature
        tempo = self._estimate_tempo(notes)
        numerator, denominator = self._estimate_time_signature(notes, tempo)
        
        # Create MIDI
        midi_data = self._create_midi(notes, tempo, numerator, denominator)
        
        return {
            'notes': notes,
            'tempo': tempo,
            'time_signature': (numerator, denominator),
            'midi_data': midi_data
        }
    
    def _process_predictions(
        self,
        all_onsets: List[Tuple[np.ndarray, float]],
        all_pitches: List[Tuple[np.ndarray, float]],
        all_durations: List[Tuple[np.ndarray, float]],
        all_velocities: List[Tuple[np.ndarray, float]]
    ) -> List[Dict[str, Any]]:
        """
        Process model predictions to create note events.
        
        Args:
            all_onsets: List of (onset predictions, start time) tuples
            all_pitches: List of (pitch predictions, start time) tuples
            all_durations: List of (duration predictions, start time) tuples
            all_velocities: List of (velocity predictions, start time) tuples
            
        Returns:
            List[Dict[str, Any]]: List of note events
        """
        notes = []
        
        # Process each segment
        for i in range(len(all_onsets)):
            onsets, start_time = all_onsets[i]
            pitch_probs, _ = all_pitches[i]
            durations, _ = all_durations[i]
            velocities, _ = all_velocities[i]
            
            # Find onset frames
            onset_frames = np.where(onsets > self.onset_threshold)[0]
            
            # Process each onset
            for frame in onset_frames:
                # Get pitch probabilities for this frame
                pitch_prob = pitch_probs[frame]
                
                # Find pitches above threshold (for polyphonic detection)
                pitch_indices = np.where(pitch_prob > self.pitch_threshold)[0]
                
                # Get duration for this frame (in frames)
                duration_frames = durations[frame]
                
                # Convert to seconds
                duration_seconds = duration_frames * self.hop_length / self.sample_rate
                
                # Get velocity for this frame
                velocity = int(velocities[frame])
                
                # Calculate onset time
                onset_time = start_time + frame * self.hop_length / self.sample_rate
                
                # Create note events for each detected pitch
                for pitch in pitch_indices:
                    note = {
                        'pitch': int(pitch),
                        'onset_time': onset_time,
                        'duration_time': duration_seconds,
                        'offset_time': onset_time + duration_seconds,
                        'velocity': velocity
                    }
                    notes.append(note)
        
        # Sort notes by onset time
        notes.sort(key=lambda x: x['onset_time'])
        
        # Merge overlapping notes with the same pitch
        merged_notes = []
        current_notes = {}  # pitch -> note
        
        for note in notes:
            pitch = note['pitch']
            
            if pitch in current_notes:
                current_note = current_notes[pitch]
                
                # If this note starts before the current note ends, merge them
                if note['onset_time'] <= current_note['offset_time']:
                    # Extend the duration
                    current_note['offset_time'] = max(current_note['offset_time'], note['offset_time'])
                    current_note['duration_time'] = current_note['offset_time'] - current_note['onset_time']
                    # Use the maximum velocity
                    current_note['velocity'] = max(current_note['velocity'], note['velocity'])
                else:
                    # Current note is finished, add to merged notes
                    merged_notes.append(current_note)
                    current_notes[pitch] = note
            else:
                current_notes[pitch] = note
        
        # Add remaining notes
        merged_notes.extend(current_notes.values())
        
        # Sort again by onset time
        merged_notes.sort(key=lambda x: x['onset_time'])
        
        return merged_notes
    
    def _estimate_tempo(self, notes: List[Dict[str, Any]]) -> float:
        """
        Estimate tempo from note events.
        
        Args:
            notes (List[Dict[str, Any]]): List of note events
            
        Returns:
            float: Estimated tempo in BPM
        """
        if not notes:
            return 120.0  # Default tempo
        
        # Extract onset times
        onset_times = np.array([note['onset_time'] for note in notes])
        
        # Calculate inter-onset intervals
        iois = np.diff(onset_times)
        
        # Filter out very short or very long IOIs
        valid_iois = iois[(iois > 0.05) & (iois < 2.0)]
        
        if len(valid_iois) == 0:
            return 120.0  # Default tempo
        
        # Find the most common IOI using a histogram
        hist, bin_edges = np.histogram(valid_iois, bins=50)
        most_common_ioi_idx = np.argmax(hist)
        most_common_ioi = (bin_edges[most_common_ioi_idx] + bin_edges[most_common_ioi_idx + 1]) / 2
        
        # Convert IOI to BPM
        tempo = 60.0 / most_common_ioi
        
        # Adjust tempo to a reasonable range (40-240 BPM)
        while tempo < 40:
            tempo *= 2
        while tempo > 240:
            tempo /= 2
        
        return tempo
    
    def _estimate_time_signature(
        self,
        notes: List[Dict[str, Any]],
        tempo: float
    ) -> Tuple[int, int]:
        """
        Estimate time signature from note events.
        
        Args:
            notes (List[Dict[str, Any]]): List of note events
            tempo (float): Estimated tempo in BPM
            
        Returns:
            Tuple[int, int]: Estimated time signature as (numerator, denominator)
        """
        if not notes:
            return (4, 4)  # Default time signature
        
        # Extract onset times
        onset_times = np.array([note['onset_time'] for note in notes])
        
        # Calculate beat duration in seconds
        beat_duration = 60.0 / tempo
        
        # Convert onset times to beats
        onset_beats = onset_times / beat_duration
        
        # Round to nearest beat
        onset_beats_rounded = np.round(onset_beats)
        
        # Calculate the difference between actual and rounded beat positions
        beat_diff = np.abs(onset_beats - onset_beats_rounded)
        
        # If the average difference is large, we might have a compound meter
        if np.mean(beat_diff) > 0.2:
            return (6, 8)  # Common compound meter
        
        # Try to detect patterns in the beat sequence
        if len(onset_beats_rounded) > 8:
            # Calculate intervals between consecutive beats
            beat_intervals = np.diff(onset_beats_rounded)
            
            # Count occurrences of each interval
            unique_intervals, counts = np.unique(beat_intervals, return_counts=True)
            
            # If we have a clear pattern of strong beats every 3 or 4 beats
            if 3 in unique_intervals and counts[np.where(unique_intervals == 3)[0][0]] > len(beat_intervals) / 10:
                return (3, 4)  # 3/4 time
            elif 4 in unique_intervals and counts[np.where(unique_intervals == 4)[0][0]] > len(beat_intervals) / 10:
                return (4, 4)  # 4/4 time
        
        # Default to 4/4 if no clear pattern is detected
        return (4, 4)
    
    def _create_midi(
        self,
        notes: List[Dict[str, Any]],
        tempo: float,
        numerator: int,
        denominator: int
    ) -> pretty_midi.PrettyMIDI:
        """
        Create MIDI file from note events.
        
        Args:
            notes (List[Dict[str, Any]]): List of note events
            tempo (float): Tempo in BPM
            numerator (int): Time signature numerator
            denominator (int): Time signature denominator
            
        Returns:
            pretty_midi.PrettyMIDI: MIDI data
        """
        # Create PrettyMIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        # Create instrument (default to piano)
        instrument = pretty_midi.Instrument(program=0)
        
        # Add notes
        for note_data in notes:
            note = pretty_midi.Note(
                velocity=min(127, max(1, note_data['velocity'])),
                pitch=note_data['pitch'],
                start=note_data['onset_time'],
                end=note_data['offset_time']
            )
            instrument.notes.append(note)
        
        # Add instrument to MIDI
        midi.instruments.append(instrument)
        
        # Add time signature
        midi.time_signature_changes.append(
            pretty_midi.TimeSignature(numerator, denominator, 0)
        )
        
        return midi
    
    def save_midi(self, midi_data: pretty_midi.PrettyMIDI, output_path: str) -> None:
        """
        Save MIDI data to file.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): MIDI data
            output_path (str): Path to save MIDI file
        """
        midi_data.write(output_path)
        print(f"MIDI file saved to {output_path}")
    
    def visualize_transcription(
        self,
        notes: List[Dict[str, Any]],
        audio_data: np.ndarray = None,
        output_path: str = None
    ) -> None:
        """
        Visualize transcription results.
        
        Args:
            notes (List[Dict[str, Any]]): List of note events
            audio_data (np.ndarray, optional): Audio time series
            output_path (str, optional): Path to save visualization
        """
        plt.figure(figsize=(12, 8))
        
        # Plot piano roll
        plt.subplot(2, 1, 1)
        
        # Create piano roll matrix
        if notes:
            max_time = max(note['offset_time'] for note in notes)
            time_resolution = 0.01  # 10ms
            num_time_steps = int(max_time / time_resolution) + 1
            piano_roll = np.zeros((128, num_time_steps))
            
            for note in notes:
                start_idx = int(note['onset_time'] / time_resolution)
                end_idx = int(note['offset_time'] / time_resolution)
                pitch = note['pitch']
                velocity = note['velocity'] / 127.0  # Normalize to [0, 1]
                
                if start_idx < num_time_steps and pitch < 128:
                    piano_roll[pitch, start_idx:end_idx] = velocity
            
            # Create custom colormap (black for no note, blue to red for velocity)
            cmap = LinearSegmentedColormap.from_list(
                'piano_roll',
                [(0, 0, 0), (0, 0, 1), (1, 0, 0)]
            )
            
            plt.imshow(
                piano_roll,
                aspect='auto',
                origin='lower',
                cmap=cmap,
                extent=[0, max_time, 0, 127]
            )
            
            plt.colorbar(label='Velocity')
            plt.ylabel('MIDI Pitch')
            plt.title('Piano Roll')
        else:
            plt.text(0.5, 0.5, 'No notes detected', ha='center', va='center')
            plt.ylabel('MIDI Pitch')
            plt.title('Piano Roll')
        
        # Plot waveform if available
        if audio_data is not None:
            plt.subplot(2, 1, 2)
            
            # Time axis
            time = np.arange(len(audio_data)) / self.sample_rate
            
            plt.plot(time, audio_data)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Waveform')
            
            # Add vertical lines for note onsets
            for note in notes:
                plt.axvline(x=note['onset_time'], color='r', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()


class RealTimeTranscriber:
    """
    Class for real-time transcription from microphone input.
    
    Attributes:
        transcriber (MusicTranscriber): Instance of MusicTranscriber
        live_audio (LiveAudioHandler): Instance of LiveAudioHandler for live audio input
        buffer_duration (float): Duration of audio buffer in seconds
        hop_duration (float): Duration between processing steps in seconds
        is_running (bool): Whether the transcriber is running
        audio_buffer (np.ndarray): Buffer for audio data
        notes_buffer (List[Dict[str, Any]]): Buffer for detected notes
        visualization_queue (Queue): Queue for visualization data
    """
    
    def __init__(
        self,
        transcriber: MusicTranscriber,
        buffer_duration: float = 5.0,
        hop_duration: float = 0.5,
        device_index: Optional[int] = None
    ):
        """
        Initialize the RealTimeTranscriber with specified parameters.
        
        Args:
            transcriber (MusicTranscriber): Instance of MusicTranscriber
            buffer_duration (float): Duration of audio buffer in seconds
            hop_duration (float): Duration between processing steps in seconds
            device_index (int, optional): Index of input device to use
        """
        self.transcriber = transcriber
        self.buffer_duration = buffer_duration
        self.hop_duration = hop_duration
        
        # Create live audio handler
        self.live_audio = LiveAudioHandler(
            sample_rate=transcriber.sample_rate,
            chunk_size=1024,
            channels=1,
            format_type=pyaudio.paFloat32,
            device_index=device_index
        )
        
        # Initialize buffers
        self.is_running = False
        self.audio_buffer = np.zeros(int(buffer_duration * transcriber.sample_rate))
        self.notes_buffer = []
        
        # Queue for visualization data
        self.visualization_queue = Queue()
        
        # Processing thread
        self.processing_thread = None
    
    def start(self, visualize: bool = False) -> None:
        """
        Start real-time transcription.
        
        Args:
            visualize (bool): Whether to visualize transcription results
        """
        if self.is_running:
            print("Real-time transcription is already running")
            return
        
        self.is_running = True
        
        # Start audio stream
        self.live_audio.start_stream()
        
        # Start processing thread
        self.processing_thread = Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start visualization thread if requested
        if visualize:
            visualization_thread = Thread(target=self._visualization_loop)
            visualization_thread.daemon = True
            visualization_thread.start()
        
        print("Real-time transcription started")
    
    def stop(self) -> List[Dict[str, Any]]:
        """
        Stop real-time transcription.
        
        Returns:
            List[Dict[str, Any]]: List of detected notes
        """
        if not self.is_running:
            print("Real-time transcription is not running")
            return self.notes_buffer
        
        self.is_running = False
        
        # Stop audio stream
        self.live_audio.stop_stream()
        
        # Wait for processing thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        print("Real-time transcription stopped")
        
        return self.notes_buffer
    
    def _processing_loop(self) -> None:
        """Processing loop for real-time transcription."""
        buffer_samples = int(self.buffer_duration * self.transcriber.sample_rate)
        hop_samples = int(self.hop_duration * self.transcriber.sample_rate)
        
        while self.is_running:
            # Get audio data from queue
            if not self.live_audio.audio_queue.empty():
                # Get the latest audio chunk
                audio_chunk = self.live_audio.audio_queue.get()
                
                # Convert bytes to numpy array
                chunk_np = np.frombuffer(audio_chunk, dtype=np.float32)
                
                # Shift buffer and add new chunk
                self.audio_buffer = np.roll(self.audio_buffer, -len(chunk_np))
                self.audio_buffer[-len(chunk_np):] = chunk_np
                
                # Process buffer every hop_duration seconds
                if self.live_audio.audio_queue.qsize() % (hop_samples // len(chunk_np)) == 0:
                    # Transcribe audio buffer
                    result = self.transcriber.transcribe_audio(
                        self.audio_buffer,
                        segment_duration=self.buffer_duration,
                        overlap=0.0
                    )
                    
                    # Update notes buffer
                    self.notes_buffer = result['notes']
                    
                    # Put data in visualization queue
                    if not self.visualization_queue.full():
                        self.visualization_queue.put({
                            'audio': self.audio_buffer.copy(),
                            'notes': self.notes_buffer.copy()
                        })
            
            # Sleep to avoid busy waiting
            time.sleep(0.01)
    
    def _visualization_loop(self) -> None:
        """Visualization loop for real-time transcription."""
        plt.figure(figsize=(12, 8))
        plt.ion()  # Interactive mode
        
        while self.is_running:
            if not self.visualization_queue.empty():
                data = self.visualization_queue.get()
                audio = data['audio']
                notes = data['notes']
                
                plt.clf()
                
                # Plot piano roll
                plt.subplot(2, 1, 1)
                
                # Create piano roll matrix
                if notes:
                    max_time = self.buffer_duration
                    time_resolution = 0.01  # 10ms
                    num_time_steps = int(max_time / time_resolution) + 1
                    piano_roll = np.zeros((128, num_time_steps))
                    
                    for note in notes:
                        start_idx = int(note['onset_time'] / time_resolution)
                        end_idx = int(note['offset_time'] / time_resolution)
                        pitch = note['pitch']
                        velocity = note['velocity'] / 127.0  # Normalize to [0, 1]
                        
                        if start_idx < num_time_steps and pitch < 128:
                            piano_roll[pitch, start_idx:end_idx] = velocity
                    
                    # Create custom colormap (black for no note, blue to red for velocity)
                    cmap = LinearSegmentedColormap.from_list(
                        'piano_roll',
                        [(0, 0, 0), (0, 0, 1), (1, 0, 0)]
                    )
                    
                    plt.imshow(
                        piano_roll,
                        aspect='auto',
                        origin='lower',
                        cmap=cmap,
                        extent=[0, max_time, 0, 127]
                    )
                    
                    plt.colorbar(label='Velocity')
                    plt.ylabel('MIDI Pitch')
                    plt.title('Piano Roll (Real-time)')
                else:
                    plt.text(0.5, 0.5, 'No notes detected', ha='center', va='center')
                    plt.ylabel('MIDI Pitch')
                    plt.title('Piano Roll (Real-time)')
                
                # Plot waveform
                plt.subplot(2, 1, 2)
                
                # Time axis
                time = np.arange(len(audio)) / self.transcriber.sample_rate
                
                plt.plot(time, audio)
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.title('Waveform (Real-time)')
                
                # Add vertical lines for note onsets
                for note in notes:
                    plt.axvline(x=note['onset_time'], color='r', alpha=0.3)
                
                plt.tight_layout()
                plt.pause(0.1)
            
            # Sleep to avoid busy waiting
            time.sleep(0.1)
        
        plt.close()
    
    def list_audio_devices(self) -> List[Dict[str, Any]]:
        """
        List available audio input devices.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries with device information
        """
        return self.live_audio.list_devices()
    
    def save_recording(self, output_path: str) -> None:
        """
        Save the current audio buffer to a file.
        
        Args:
            output_path (str): Path to save audio file
        """
        self.live_audio.save_recording(output_path, self.audio_buffer)
        print(f"Recording saved to {output_path}")
    
    def save_transcription(self, output_path: str) -> None:
        """
        Save the current transcription to a MIDI file.
        
        Args:
            output_path (str): Path to save MIDI file
        """
        if not self.notes_buffer:
            print("No notes to save")
            return
        
        # Estimate tempo and time signature
        tempo = self.transcriber._estimate_tempo(self.notes_buffer)
        numerator, denominator = self.transcriber._estimate_time_signature(self.notes_buffer, tempo)
        
        # Create MIDI
        midi_data = self.transcriber._create_midi(
            self.notes_buffer,
            tempo,
            numerator,
            denominator
        )
        
        # Save MIDI
        self.transcriber.save_midi(midi_data, output_path)


def load_model(
    checkpoint_path: str,
    model_config: Optional[Dict[str, Any]] = None,
    device: torch.device = None
) -> MusicTranscriptionModel:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model_config (Dict[str, Any], optional): Model configuration
        device (torch.device, optional): Device to use for inference
        
    Returns:
        MusicTranscriptionModel: Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Default model configuration
    if model_config is None:
        model_config = {
            'input_channels': 1,
            'hidden_channels': [32, 64, 128, 256],
            'lstm_hidden_size': 256,
            'attention_dim': 128,
            'num_pitches': 128,
            'max_duration_frames': 100,
            'num_instruments': 10,
            'dropout': 0.0  # No dropout for inference
        }
    
    # Create model
    model = MusicTranscriptionModel(**model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Music transcription inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, help="Path to input audio file")
    parser.add_argument("--output", type=str, help="Path to output MIDI file")
    parser.add_argument("--visualize", action="store_true", help="Visualize transcription")
    parser.add_argument("--real_time", action="store_true", help="Use real-time transcription")
    parser.add_argument("--list_devices", action="store_true", help="List audio input devices")
    parser.add_argument("--device_index", type=int, help="Index of audio input device to use")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate for audio processing")
    parser.add_argument("--onset_threshold", type=float, default=0.5, help="Threshold for onset detection")
    parser.add_argument("--pitch_threshold", type=float, default=0.5, help="Threshold for pitch detection")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Create transcriber
    transcriber = MusicTranscriber(
        model=model,
        sample_rate=args.sample_rate,
        onset_threshold=args.onset_threshold,
        pitch_threshold=args.pitch_threshold
    )
    
    if args.list_devices:
        # Create real-time transcriber
        rt_transcriber = RealTimeTranscriber(transcriber)
        
        # List audio devices
        devices = rt_transcriber.list_audio_devices()
        print("Available audio input devices:")
        for device in devices:
            print(f"Index: {device['index']}, Name: {device['name']}, "
                  f"Channels: {device['channels']}, Sample Rate: {device['sample_rate']}")
    
    elif args.real_time:
        # Create real-time transcriber
        rt_transcriber = RealTimeTranscriber(
            transcriber=transcriber,
            device_index=args.device_index
        )
        
        try:
            # Start real-time transcription
            rt_transcriber.start(visualize=args.visualize)
            
            print("Press Enter to stop...")
            input()
            
            # Stop real-time transcription
            notes = rt_transcriber.stop()
            
            # Save transcription if output path is provided
            if args.output:
                rt_transcriber.save_transcription(args.output)
        
        except KeyboardInterrupt:
            # Stop real-time transcription
            rt_transcriber.stop()
    
    elif args.input:
        # Transcribe audio file
        result = transcriber.transcribe_file(args.input)
        
        # Print transcription summary
        print(f"Transcription completed!")
        print(f"Detected {len(result['notes'])} notes")
        print(f"Estimated tempo: {result['tempo']:.1f} BPM")
        print(f"Estimated time signature: {result['time_signature'][0]}/{result['time_signature'][1]}")
        
        # Save MIDI if output path is provided
        if args.output:
            transcriber.save_midi(result['midi_data'], args.output)
        
        # Visualize if requested
        if args.visualize:
            # Load audio for visualization
            audio_data, _ = transcriber.audio_loader.load_file(args.input)
            
            # Visualize transcription
            vis_output = args.output.replace('.mid', '.png') if args.output else None
            transcriber.visualize_transcription(result['notes'], audio_data, vis_output)
    
    else:
        print("Please provide an input audio file or use real-time transcription")
