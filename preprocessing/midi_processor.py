"""
MIDI Processor Module for Music Transcription System

This module provides functionality for processing MIDI files and handling MIDI data.
It includes classes and functions for loading MIDI files, extracting note events,
and converting between audio and MIDI representations.

Classes:
    MIDILoader: Handles loading and basic processing of MIDI files
    MIDIConverter: Converts between audio features and MIDI representations
    NoteEventExtractor: Extracts note events from MIDI data
"""

import os
import numpy as np
import mido
import pretty_midi
import librosa
from typing import Dict, List, Tuple, Optional, Union, Any


class MIDILoader:
    """
    Class for loading and basic processing of MIDI files.
    
    Attributes:
        skip_empty_tracks (bool): Whether to skip empty tracks
    """
    
    def __init__(self, skip_empty_tracks: bool = True):
        """
        Initialize the MIDILoader with specified parameters.
        
        Args:
            skip_empty_tracks (bool): Whether to skip empty tracks
        """
        self.skip_empty_tracks = skip_empty_tracks
    
    def load_file(self, file_path: str) -> mido.MidiFile:
        """
        Load a MIDI file and return the MIDI object.
        
        Args:
            file_path (str): Path to the MIDI file
            
        Returns:
            mido.MidiFile: MIDI file object
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MIDI file not found: {file_path}")
        
        try:
            midi_file = mido.MidiFile(file_path)
            return midi_file
        except Exception as e:
            raise ValueError(f"Error loading MIDI file: {e}")
    
    def load_file_pretty_midi(self, file_path: str) -> pretty_midi.PrettyMIDI:
        """
        Load a MIDI file using pretty_midi and return the PrettyMIDI object.
        
        Args:
            file_path (str): Path to the MIDI file
            
        Returns:
            pretty_midi.PrettyMIDI: PrettyMIDI object
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MIDI file not found: {file_path}")
        
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path)
            return midi_data
        except Exception as e:
            raise ValueError(f"Error loading MIDI file with pretty_midi: {e}")
    
    def get_tempo(self, midi_file: mido.MidiFile) -> float:
        """
        Extract tempo from a MIDI file.
        
        Args:
            midi_file (mido.MidiFile): MIDI file object
            
        Returns:
            float: Tempo in BPM (beats per minute)
        """
        # Default tempo (120 BPM)
        default_tempo = 500000  # microseconds per beat
        
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    # Convert microseconds per beat to BPM
                    return 60 * 1000000 / msg.tempo
        
        # Return default tempo if no tempo message found
        return 60 * 1000000 / default_tempo
    
    def get_time_signature(self, midi_file: mido.MidiFile) -> Tuple[int, int]:
        """
        Extract time signature from a MIDI file.
        
        Args:
            midi_file (mido.MidiFile): MIDI file object
            
        Returns:
            Tuple[int, int]: Time signature as (numerator, denominator)
        """
        # Default time signature (4/4)
        numerator, denominator = 4, 4
        
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'time_signature':
                    numerator = msg.numerator
                    denominator = msg.denominator
                    return numerator, denominator
        
        # Return default time signature if no time signature message found
        return numerator, denominator
    
    def get_key_signature(self, midi_file: mido.MidiFile) -> str:
        """
        Extract key signature from a MIDI file.
        
        Args:
            midi_file (mido.MidiFile): MIDI file object
            
        Returns:
            str: Key signature
        """
        # MIDI key signatures are represented as -7 to +7 (flats to sharps)
        key_signatures = {
            -7: 'Cb',
            -6: 'Gb',
            -5: 'Db',
            -4: 'Ab',
            -3: 'Eb',
            -2: 'Bb',
            -1: 'F',
            0: 'C',
            1: 'G',
            2: 'D',
            3: 'A',
            4: 'E',
            5: 'B',
            6: 'F#',
            7: 'C#'
        }
        
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'key_signature':
                    key = msg.key
                    if isinstance(key, int):
                        return key_signatures.get(key, 'Unknown')
                    return key
        
        # Return default key signature if no key signature message found
        return 'C'
    
    def save_midi(self, midi_data: Union[mido.MidiFile, pretty_midi.PrettyMIDI], file_path: str) -> None:
        """
        Save MIDI data to a file.
        
        Args:
            midi_data (Union[mido.MidiFile, pretty_midi.PrettyMIDI]): MIDI data to save
            file_path (str): Path where to save the MIDI file
            
        Raises:
            ValueError: If there's an error saving the file
        """
        try:
            if isinstance(midi_data, mido.MidiFile):
                midi_data.save(file_path)
            elif isinstance(midi_data, pretty_midi.PrettyMIDI):
                midi_data.write(file_path)
            else:
                raise ValueError("Unsupported MIDI data type")
        except Exception as e:
            raise ValueError(f"Error saving MIDI file: {e}")


class NoteEventExtractor:
    """
    Class for extracting note events from MIDI data.
    
    Attributes:
        ticks_per_beat (int): Number of ticks per beat
    """
    
    def __init__(self, ticks_per_beat: int = 480):
        """
        Initialize the NoteEventExtractor with specified parameters.
        
        Args:
            ticks_per_beat (int): Number of ticks per beat
        """
        self.ticks_per_beat = ticks_per_beat
    
    def extract_notes_from_midi(self, midi_file: mido.MidiFile) -> List[Dict[str, Any]]:
        """
        Extract note events from a MIDI file.
        
        Args:
            midi_file (mido.MidiFile): MIDI file object
            
        Returns:
            List[Dict[str, Any]]: List of note events with onset, offset, pitch, velocity
        """
        notes = []
        
        # Process each track
        for track_idx, track in enumerate(midi_file.tracks):
            # Keep track of active notes (note_on events without corresponding note_off)
            active_notes = {}
            
            # Absolute time in ticks
            abs_time = 0
            
            for msg in track:
                # Update absolute time
                abs_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Note on event
                    note_id = (msg.channel, msg.note)
                    active_notes[note_id] = {
                        'track': track_idx,
                        'channel': msg.channel,
                        'pitch': msg.note,
                        'velocity': msg.velocity,
                        'onset_ticks': abs_time,
                        'onset_time': None,  # Will be calculated later
                        'offset_ticks': None,
                        'offset_time': None,
                        'duration_ticks': None,
                        'duration_time': None
                    }
                
                elif (msg.type == 'note_off' or 
                      (msg.type == 'note_on' and msg.velocity == 0)):
                    # Note off event
                    note_id = (msg.channel, msg.note)
                    if note_id in active_notes:
                        note = active_notes[note_id]
                        note['offset_ticks'] = abs_time
                        note['duration_ticks'] = note['offset_ticks'] - note['onset_ticks']
                        notes.append(note)
                        del active_notes[note_id]
        
        # Convert ticks to seconds
        tempo = 500000  # Default tempo (microseconds per beat)
        seconds_per_tick = tempo / (1000000 * midi_file.ticks_per_beat)
        
        for note in notes:
            note['onset_time'] = note['onset_ticks'] * seconds_per_tick
            note['offset_time'] = note['offset_ticks'] * seconds_per_tick
            note['duration_time'] = note['duration_ticks'] * seconds_per_tick
        
        # Sort notes by onset time
        notes.sort(key=lambda x: x['onset_time'])
        
        return notes
    
    def extract_notes_from_pretty_midi(self, midi_data: pretty_midi.PrettyMIDI) -> List[Dict[str, Any]]:
        """
        Extract note events from a PrettyMIDI object.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): PrettyMIDI object
            
        Returns:
            List[Dict[str, Any]]: List of note events with onset, offset, pitch, velocity
        """
        notes = []
        
        # Process each instrument
        for inst_idx, instrument in enumerate(midi_data.instruments):
            for note in instrument.notes:
                notes.append({
                    'instrument': inst_idx,
                    'program': instrument.program,
                    'is_drum': instrument.is_drum,
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'onset_time': note.start,
                    'offset_time': note.end,
                    'duration_time': note.end - note.start
                })
        
        # Sort notes by onset time
        notes.sort(key=lambda x: x['onset_time'])
        
        return notes
    
    def get_piano_roll(self, midi_data: pretty_midi.PrettyMIDI, fs: int = 100) -> np.ndarray:
        """
        Get piano roll representation from a PrettyMIDI object.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): PrettyMIDI object
            fs (int): Sampling frequency of the piano roll
            
        Returns:
            np.ndarray: Piano roll matrix
        """
        return midi_data.get_piano_roll(fs=fs)
    
    def get_onsets(self, notes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract onset times from note events.
        
        Args:
            notes (List[Dict[str, Any]]): List of note events
            
        Returns:
            np.ndarray: Array of onset times in seconds
        """
        return np.array([note['onset_time'] for note in notes])
    
    def get_offsets(self, notes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract offset times from note events.
        
        Args:
            notes (List[Dict[str, Any]]): List of note events
            
        Returns:
            np.ndarray: Array of offset times in seconds
        """
        return np.array([note['offset_time'] for note in notes])
    
    def get_pitches(self, notes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract pitches from note events.
        
        Args:
            notes (List[Dict[str, Any]]): List of note events
            
        Returns:
            np.ndarray: Array of MIDI pitches
        """
        return np.array([note['pitch'] for note in notes])
    
    def get_velocities(self, notes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract velocities from note events.
        
        Args:
            notes (List[Dict[str, Any]]): List of note events
            
        Returns:
            np.ndarray: Array of MIDI velocities
        """
        return np.array([note['velocity'] for note in notes])


class MIDIConverter:
    """
    Class for converting between audio features and MIDI representations.
    
    Attributes:
        sample_rate (int): Sample rate of the audio data
        hop_length (int): Number of samples between successive frames
    """
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        """
        Initialize the MIDIConverter with specified parameters.
        
        Args:
            sample_rate (int): Sample rate of the audio data
            hop_length (int): Number of samples between successive frames
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def frames_to_time(self, frames: np.ndarray) -> np.ndarray:
        """
        Convert frame indices to time (seconds).
        
        Args:
            frames (np.ndarray): Frame indices
            
        Returns:
            np.ndarray: Time in seconds
        """
        return librosa.frames_to_time(frames, sr=self.sample_rate, hop_length=self.hop_length)
    
    def time_to_frames(self, times: np.ndarray) -> np.ndarray:
        """
        Convert time (seconds) to frame indices.
        
        Args:
            times (np.ndarray): Time in seconds
            
        Returns:
            np.ndarray: Frame indices
        """
        return librosa.time_to_frames(times, sr=self.sample_rate, hop_length=self.hop_length)
    
    def hz_to_midi(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Convert frequencies (Hz) to MIDI note numbers.
        
        Args:
            frequencies (np.ndarray): Frequencies in Hz
            
        Returns:
            np.ndarray: MIDI note numbers
        """
        return librosa.hz_to_midi(frequencies)
    
    def midi_to_hz(self, notes: np.ndarray) -> np.ndarray:
        """
        Convert MIDI note numbers to frequencies (Hz).
        
        Args:
            notes (np.ndarray): MIDI note numbers
            
        Returns:
            np.ndarray: Frequencies in Hz
        """
        return librosa.midi_to_hz(notes)
    
    def create_midi_from_notes(
        self, 
        onsets: np.ndarray, 
        pitches: np.ndarray, 
        durations: np.ndarray, 
        velocities: np.ndarray = None,
        instrument: int = 0,
        is_drum: bool = False,
        tempo: float = 120.0
    ) -> pretty_midi.PrettyMIDI:
        """
        Create a PrettyMIDI object from note events.
        
        Args:
            onsets (np.ndarray): Note onset times in seconds
            pitches (np.ndarray): MIDI note numbers
            durations (np.ndarray): Note durations in seconds
            velocities (np.ndarray, optional): MIDI velocities (0-127)
            instrument (int): MIDI program number
            is_drum (bool): Whether the instrument is a drum kit
            tempo (float): Tempo in BPM
            
        Returns:
            pretty_midi.PrettyMIDI: PrettyMIDI object
        """
        if velocities is None:
            velocities = np.ones_like(onsets) * 100  # Default velocity
        
        # Create PrettyMIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        # Create instrument
        inst = pretty_midi.Instrument(program=instrument, is_drum=is_drum)
        
        # Add notes
        for i in range(len(onsets)):
            note = pretty_midi.Note(
                velocity=int(velocities[i]),
                pitch=int(pitches[i]),
                start=float(onsets[i]),
                end=float(onsets[i] + durations[i])
            )
            inst.notes.append(note)
        
        # Add instrument to MIDI
        midi.instruments.append(inst)
        
        return midi
    
    def create_midi_from_piano_roll(
        self, 
        piano_roll: np.ndarray, 
        fs: int = 100,
        instrument: int = 0,
        is_drum: bool = False,
        tempo: float = 120.0
    ) -> pretty_midi.PrettyMIDI:
        """
        Create a PrettyMIDI object from a piano roll matrix.
        
        Args:
            piano_roll (np.ndarray): Piano roll matrix (128 x frames)
            fs (int): Sampling frequency of the piano roll
            instrument (int): MIDI program number
            is_drum (bool): Whether the instrument is a drum kit
            tempo (float): Tempo in BPM
            
        Returns:
            pretty_midi.PrettyMIDI: PrettyMIDI object
        """
        # Create PrettyMIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        # Create instrument
        inst = pretty_midi.Instrument(program=instrument, is_drum=is_drum)
        
        # Convert piano roll to notes
        piano_roll = piano_roll.T  # Transpose to (frames x 128)
        
        # Find note onsets and offsets
        onset_frames = np.where(np.diff(piano_roll > 0, axis=0) > 0)
        offset_frames = np.where(np.diff(piano_roll > 0, axis=0) < 0)
        
        # Group by pitch
        for pitch in range(128):
            # Find onsets for this pitch
            pitch_onsets = onset_frames[0][onset_frames[1] == pitch]
            
            # Find offsets for this pitch
            pitch_offsets = offset_frames[0][offset_frames[1] == pitch]
            
            # Match onsets with offsets
            for onset in pitch_onsets:
                # Find the next offset after this onset
                offset_candidates = pitch_offsets[pitch_offsets > onset]
                
                if len(offset_candidates) > 0:
                    offset = offset_candidates[0]
                else:
                    # If no offset found, use the end of the piano roll
                    offset = piano_roll.shape[0] - 1
                
                # Get velocity from piano roll
                velocity = int(np.max(piano_roll[onset:offset+1, pitch]) * 127)
                
                # Create note
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=onset / fs,
                    end=offset / fs
                )
                inst.notes.append(note)
        
        # Add instrument to MIDI
        midi.instruments.append(inst)
        
        return midi
