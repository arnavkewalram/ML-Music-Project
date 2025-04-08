"""
Output Format Converters Module for Music Transcription System

This module provides functionality for converting MIDI data to various output formats:
- MusicXML
- PDF sheet music
- LilyPond
- MuseScore
- Piano roll visualization
- Audio playback

Classes:
    MIDIToMusicXML: Converts MIDI to MusicXML format
    MIDIToPDF: Converts MIDI to PDF sheet music
    MIDIToLilyPond: Converts MIDI to LilyPond format
    MIDIToMuseScore: Converts MIDI to MuseScore format
    MIDIToPianoRoll: Creates piano roll visualization from MIDI
    MIDIToAudio: Converts MIDI to audio for playback
"""

import os
import subprocess
import tempfile
import numpy as np
import pretty_midi
import music21
from music21 import converter, stream, note, chord, meter, tempo, midi
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional, Union, Any
import io
import base64
from PIL import Image


class MIDIToMusicXML:
    """
    Class for converting MIDI to MusicXML format.
    
    Methods:
        convert: Convert MIDI data to MusicXML
        save: Save MusicXML to file
    """
    
    @staticmethod
    def convert(midi_data: pretty_midi.PrettyMIDI) -> music21.stream.Score:
        """
        Convert MIDI data to MusicXML using music21.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): MIDI data
            
        Returns:
            music21.stream.Score: MusicXML score
        """
        # Create a temporary MIDI file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_midi:
            midi_path = temp_midi.name
            midi_data.write(midi_path)
        
        try:
            # Convert MIDI to music21 score
            score = converter.parse(midi_path)
            
            # Add time signature if not present
            if not score.getTimeSignatures():
                # Get time signature from MIDI
                if midi_data.time_signature_changes:
                    ts = midi_data.time_signature_changes[0]
                    time_signature = meter.TimeSignature(f"{ts.numerator}/{ts.denominator}")
                else:
                    # Default to 4/4
                    time_signature = meter.TimeSignature('4/4')
                
                score.insert(0, time_signature)
            
            # Add tempo if not present
            if not score.getTempos():
                # Get tempo from MIDI
                if midi_data.get_tempo_changes()[1].size > 0:
                    bpm = midi_data.get_tempo_changes()[1][0]
                else:
                    # Default to 120 BPM
                    bpm = 120
                
                score.insert(0, tempo.MetronomeMark(number=bpm))
            
            return score
        
        finally:
            # Clean up temporary file
            os.unlink(midi_path)
    
    @staticmethod
    def save(score: music21.stream.Score, output_path: str) -> None:
        """
        Save MusicXML score to file.
        
        Args:
            score (music21.stream.Score): MusicXML score
            output_path (str): Path to save MusicXML file
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save as MusicXML
        score.write('musicxml', fp=output_path)
        print(f"MusicXML file saved to {output_path}")


class MIDIToPDF:
    """
    Class for converting MIDI to PDF sheet music.
    
    Methods:
        convert: Convert MIDI data to PDF using music21 and LilyPond
        save: Save PDF to file
    """
    
    @staticmethod
    def convert(
        midi_data: pretty_midi.PrettyMIDI,
        title: str = "Music Transcription",
        composer: str = "AI Transcription System"
    ) -> bytes:
        """
        Convert MIDI data to PDF sheet music using music21 and LilyPond.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): MIDI data
            title (str): Title of the sheet music
            composer (str): Composer name
            
        Returns:
            bytes: PDF data
        """
        # Convert MIDI to music21 score
        score = MIDIToMusicXML.convert(midi_data)
        
        # Add metadata
        score.metadata = music21.metadata.Metadata()
        score.metadata.title = title
        score.metadata.composer = composer
        
        # Create a temporary directory for output files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Path for PDF output
            pdf_path = os.path.join(temp_dir, "output.pdf")
            
            # Write PDF using LilyPond backend
            try:
                score.write('lily.pdf', fp=pdf_path)
                
                # Read PDF data
                with open(pdf_path, 'rb') as f:
                    pdf_data = f.read()
                
                return pdf_data
            
            except Exception as e:
                print(f"Error converting to PDF: {e}")
                
                # Fallback: Try using MuseScore if available
                try:
                    # Save as MusicXML
                    xml_path = os.path.join(temp_dir, "output.musicxml")
                    score.write('musicxml', fp=xml_path)
                    
                    # Convert MusicXML to PDF using MuseScore
                    subprocess.run(
                        ["mscore", "-o", pdf_path, xml_path],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Read PDF data
                    with open(pdf_path, 'rb') as f:
                        pdf_data = f.read()
                    
                    return pdf_data
                
                except Exception as e2:
                    print(f"Error using MuseScore fallback: {e2}")
                    raise ValueError("Failed to convert MIDI to PDF. Make sure LilyPond or MuseScore is installed.")
    
    @staticmethod
    def save(pdf_data: bytes, output_path: str) -> None:
        """
        Save PDF data to file.
        
        Args:
            pdf_data (bytes): PDF data
            output_path (str): Path to save PDF file
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save PDF
        with open(output_path, 'wb') as f:
            f.write(pdf_data)
        
        print(f"PDF file saved to {output_path}")


class MIDIToLilyPond:
    """
    Class for converting MIDI to LilyPond format.
    
    Methods:
        convert: Convert MIDI data to LilyPond
        save: Save LilyPond to file
    """
    
    @staticmethod
    def convert(
        midi_data: pretty_midi.PrettyMIDI,
        title: str = "Music Transcription",
        composer: str = "AI Transcription System"
    ) -> str:
        """
        Convert MIDI data to LilyPond format using music21.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): MIDI data
            title (str): Title of the sheet music
            composer (str): Composer name
            
        Returns:
            str: LilyPond code
        """
        # Convert MIDI to music21 score
        score = MIDIToMusicXML.convert(midi_data)
        
        # Add metadata
        score.metadata = music21.metadata.Metadata()
        score.metadata.title = title
        score.metadata.composer = composer
        
        # Create a temporary file for LilyPond output
        with tempfile.NamedTemporaryFile(suffix='.ly', delete=False) as temp_ly:
            ly_path = temp_ly.name
        
        try:
            # Write LilyPond file
            score.write('lily', fp=ly_path)
            
            # Read LilyPond code
            with open(ly_path, 'r') as f:
                ly_code = f.read()
            
            return ly_code
        
        finally:
            # Clean up temporary file
            os.unlink(ly_path)
    
    @staticmethod
    def save(ly_code: str, output_path: str) -> None:
        """
        Save LilyPond code to file.
        
        Args:
            ly_code (str): LilyPond code
            output_path (str): Path to save LilyPond file
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save LilyPond code
        with open(output_path, 'w') as f:
            f.write(ly_code)
        
        print(f"LilyPond file saved to {output_path}")
    
    @staticmethod
    def compile(ly_path: str, output_dir: str = None) -> str:
        """
        Compile LilyPond file to PDF.
        
        Args:
            ly_path (str): Path to LilyPond file
            output_dir (str, optional): Directory to save PDF file
            
        Returns:
            str: Path to generated PDF file
        """
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(ly_path))
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base name without extension
        base_name = os.path.splitext(os.path.basename(ly_path))[0]
        
        # Output PDF path
        pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
        
        try:
            # Compile LilyPond to PDF
            subprocess.run(
                ["lilypond", "-o", os.path.join(output_dir, base_name), ly_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            print(f"LilyPond compiled to {pdf_path}")
            return pdf_path
        
        except subprocess.CalledProcessError as e:
            print(f"Error compiling LilyPond: {e}")
            print(f"stderr: {e.stderr.decode()}")
            raise ValueError("Failed to compile LilyPond. Make sure LilyPond is installed.")


class MIDIToMuseScore:
    """
    Class for converting MIDI to MuseScore format.
    
    Methods:
        convert: Convert MIDI data to MuseScore
        save: Save MuseScore to file
    """
    
    @staticmethod
    def convert(midi_data: pretty_midi.PrettyMIDI) -> bytes:
        """
        Convert MIDI data to MuseScore format.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): MIDI data
            
        Returns:
            bytes: MuseScore file data
        """
        # Create a temporary directory for output files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save MIDI to temporary file
            midi_path = os.path.join(temp_dir, "input.mid")
            midi_data.write(midi_path)
            
            # Output MuseScore path
            mscz_path = os.path.join(temp_dir, "output.mscz")
            
            try:
                # Convert MIDI to MuseScore using MuseScore CLI
                subprocess.run(
                    ["mscore", "-o", mscz_path, midi_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Read MuseScore data
                with open(mscz_path, 'rb') as f:
                    mscz_data = f.read()
                
                return mscz_data
            
            except subprocess.CalledProcessError as e:
                print(f"Error converting to MuseScore: {e}")
                print(f"stderr: {e.stderr.decode()}")
                raise ValueError("Failed to convert MIDI to MuseScore. Make sure MuseScore is installed.")
    
    @staticmethod
    def save(mscz_data: bytes, output_path: str) -> None:
        """
        Save MuseScore data to file.
        
        Args:
            mscz_data (bytes): MuseScore data
            output_path (str): Path to save MuseScore file
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save MuseScore file
        with open(output_path, 'wb') as f:
            f.write(mscz_data)
        
        print(f"MuseScore file saved to {output_path}")


class MIDIToPianoRoll:
    """
    Class for creating piano roll visualization from MIDI.
    
    Methods:
        convert: Convert MIDI data to piano roll visualization
        save: Save piano roll visualization to file
    """
    
    @staticmethod
    def convert(
        midi_data: pretty_midi.PrettyMIDI,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        show_beats: bool = True,
        show_velocity: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Convert MIDI data to piano roll visualization.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): MIDI data
            figsize (Tuple[int, int]): Figure size
            dpi (int): DPI for rendering
            show_beats (bool): Whether to show beat markers
            show_velocity (bool): Whether to color notes by velocity
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Get piano roll
        fs = 100  # 100 Hz (10ms resolution)
        piano_roll = midi_data.get_piano_roll(fs=fs)
        
        # Determine time range
        end_time = midi_data.get_end_time()
        times = np.arange(0, end_time, 1/fs)
        
        # Create custom colormap (black for no note, blue to red for velocity)
        if show_velocity:
            cmap = LinearSegmentedColormap.from_list(
                'piano_roll',
                [(0, 0, 0), (0, 0, 1), (1, 0, 0)]
            )
        else:
            cmap = 'Blues'
        
        # Plot piano roll
        img = ax.imshow(
            piano_roll,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            extent=[0, end_time, 0, 127]
        )
        
        # Add colorbar if showing velocity
        if show_velocity:
            plt.colorbar(img, ax=ax, label='Velocity')
        
        # Add beat markers if requested
        if show_beats:
            # Get beat times
            beat_times = midi_data.get_beats()
            
            # Plot beat markers
            for beat_time in beat_times:
                ax.axvline(x=beat_time, color='g', alpha=0.3)
        
        # Add measure markers
        if midi_data.time_signature_changes:
            # Get time signature
            ts = midi_data.time_signature_changes[0]
            numerator = ts.numerator
            denominator = ts.denominator
            
            # Calculate measure duration in seconds
            tempo = midi_data.get_tempo_changes()[1][0] if midi_data.get_tempo_changes()[1].size > 0 else 120
            seconds_per_beat = 60 / tempo
            seconds_per_measure = seconds_per_beat * numerator * 4 / denominator
            
            # Plot measure markers
            measure_times = np.arange(0, end_time, seconds_per_measure)
            for measure_time in measure_times:
                ax.axvline(x=measure_time, color='r', alpha=0.5, linestyle='--')
        
        # Set labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('MIDI Pitch')
        ax.set_title('Piano Roll Visualization')
        
        # Set y-ticks to show octaves
        ax.set_yticks(np.arange(0, 128, 12))
        ax.set_yticklabels([f"C{i-1}" for i in range(11)])
        
        # Add grid
        ax.grid(alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        return fig, ax
    
    @staticmethod
    def save(fig: plt.Figure, output_path: str) -> None:
        """
        Save piano roll visualization to file.
        
        Args:
            fig (plt.Figure): Figure object
            output_path (str): Path to save image file
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save figure
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Piano roll visualization saved to {output_path}")
    
    @staticmethod
    def to_image_data(fig: plt.Figure, format: str = 'png') -> bytes:
        """
        Convert figure to image data.
        
        Args:
            fig (plt.Figure): Figure object
            format (str): Image format ('png', 'jpg', 'svg', etc.)
            
        Returns:
            bytes: Image data
        """
        # Save figure to in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format=format, bbox_inches='tight')
        buf.seek(0)
        
        # Get image data
        img_data = buf.getvalue()
        buf.close()
        
        return img_data
    
    @staticmethod
    def to_base64(fig: plt.Figure, format: str = 'png') -> str:
        """
        Convert figure to base64-encoded image.
        
        Args:
            fig (plt.Figure): Figure object
            format (str): Image format ('png', 'jpg', 'svg', etc.)
            
        Returns:
            str: Base64-encoded image data
        """
        # Get image data
        img_data = MIDIToPianoRoll.to_image_data(fig, format)
        
        # Encode as base64
        b64_data = base64.b64encode(img_data).decode('utf-8')
        
        return f"data:image/{format};base64,{b64_data}"


class MIDIToAudio:
    """
    Class for converting MIDI to audio for playback.
    
    Methods:
        convert: Convert MIDI data to audio
        save: Save audio to file
    """
    
    @staticmethod
    def convert(
        midi_data: pretty_midi.PrettyMIDI,
        sample_rate: int = 44100,
        sf2_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Convert MIDI data to audio.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): MIDI data
            sample_rate (int): Sample rate for audio
            sf2_path (str, optional): Path to SoundFont file
            
        Returns:
            np.ndarray: Audio data
        """
        # Synthesize audio
        if sf2_path and os.path.exists(sf2_path):
            # Use specified SoundFont
            midi_data.sf2_path = sf2_path
        
        audio_data = midi_data.synthesize(fs=sample_rate)
        
        return audio_data
    
    @staticmethod
    def save(
        audio_data: np.ndarray,
        output_path: str,
        sample_rate: int = 44100
    ) -> None:
        """
        Save audio data to file.
        
        Args:
            audio_data (np.ndarray): Audio data
            output_path (str): Path to save audio file
            sample_rate (int): Sample rate for audio
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Determine file format from extension
        _, ext = os.path.splitext(output_path)
        ext = ext.lower()
        
        if ext == '.wav':
            # Save as WAV
            import scipy.io.wavfile
            scipy.io.wavfile.write(output_path, sample_rate, audio_data)
        
        elif ext == '.mp3':
            # Save as MP3 using ffmpeg
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                wav_path = temp_wav.name
            
            try:
                # Save as WAV first
                import scipy.io.wavfile
                scipy.io.wavfile.write(wav_path, sample_rate, audio_data)
                
                # Convert WAV to MP3 using ffmpeg
                subprocess.run(
                    ["ffmpeg", "-i", wav_path, "-q:a", "2", output_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            finally:
                # Clean up temporary file
                os.unlink(wav_path)
        
        else:
            # Default to WAV
            import scipy.io.wavfile
            scipy.io.wavfile.write(output_path, sample_rate, audio_data)
        
        print(f"Audio file saved to {output_path}")


class FormatConverter:
    """
    Main class for converting between different formats.
    
    Methods:
        midi_to_musicxml: Convert MIDI to MusicXML
        midi_to_pdf: Convert MIDI to PDF
        midi_to_lilypond: Convert MIDI to LilyPond
        midi_to_musescore: Convert MIDI to MuseScore
        midi_to_piano_roll: Convert MIDI to piano roll visualization
        midi_to_audio: Convert MIDI to audio
        wav_to_all_formats: Convert WAV to all output formats
    """
    
    def __init__(
        self,
        transcriber = None,
        sample_rate: int = 22050,
        sf2_path: Optional[str] = None
    ):
        """
        Initialize the FormatConverter with specified parameters.
        
        Args:
            transcriber: Instance of MusicTranscriber (optional)
            sample_rate (int): Sample rate for audio processing
            sf2_path (str, optional): Path to SoundFont file
        """
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.sf2_path = sf2_path
    
    def midi_to_musicxml(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        output_path: str
    ) -> None:
        """
        Convert MIDI to MusicXML and save to file.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): MIDI data
            output_path (str): Path to save MusicXML file
        """
        # Convert MIDI to MusicXML
        score = MIDIToMusicXML.convert(midi_data)
        
        # Save MusicXML
        MIDIToMusicXML.save(score, output_path)
    
    def midi_to_pdf(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        output_path: str,
        title: str = "Music Transcription",
        composer: str = "AI Transcription System"
    ) -> None:
        """
        Convert MIDI to PDF and save to file.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): MIDI data
            output_path (str): Path to save PDF file
            title (str): Title of the sheet music
            composer (str): Composer name
        """
        # Convert MIDI to PDF
        pdf_data = MIDIToPDF.convert(midi_data, title, composer)
        
        # Save PDF
        MIDIToPDF.save(pdf_data, output_path)
    
    def midi_to_lilypond(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        output_path: str,
        title: str = "Music Transcription",
        composer: str = "AI Transcription System",
        compile_pdf: bool = False
    ) -> None:
        """
        Convert MIDI to LilyPond and save to file.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): MIDI data
            output_path (str): Path to save LilyPond file
            title (str): Title of the sheet music
            composer (str): Composer name
            compile_pdf (bool): Whether to compile LilyPond to PDF
        """
        # Convert MIDI to LilyPond
        ly_code = MIDIToLilyPond.convert(midi_data, title, composer)
        
        # Save LilyPond
        MIDIToLilyPond.save(ly_code, output_path)
        
        # Compile to PDF if requested
        if compile_pdf:
            MIDIToLilyPond.compile(output_path)
    
    def midi_to_musescore(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        output_path: str
    ) -> None:
        """
        Convert MIDI to MuseScore and save to file.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): MIDI data
            output_path (str): Path to save MuseScore file
        """
        # Convert MIDI to MuseScore
        mscz_data = MIDIToMuseScore.convert(midi_data)
        
        # Save MuseScore
        MIDIToMuseScore.save(mscz_data, output_path)
    
    def midi_to_piano_roll(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        output_path: str,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        show_beats: bool = True,
        show_velocity: bool = True
    ) -> None:
        """
        Convert MIDI to piano roll visualization and save to file.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): MIDI data
            output_path (str): Path to save image file
            figsize (Tuple[int, int]): Figure size
            dpi (int): DPI for rendering
            show_beats (bool): Whether to show beat markers
            show_velocity (bool): Whether to color notes by velocity
        """
        # Convert MIDI to piano roll
        fig, _ = MIDIToPianoRoll.convert(midi_data, figsize, dpi, show_beats, show_velocity)
        
        # Save piano roll
        MIDIToPianoRoll.save(fig, output_path)
    
    def midi_to_audio(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        output_path: str,
        sample_rate: int = 44100
    ) -> None:
        """
        Convert MIDI to audio and save to file.
        
        Args:
            midi_data (pretty_midi.PrettyMIDI): MIDI data
            output_path (str): Path to save audio file
            sample_rate (int): Sample rate for audio
        """
        # Convert MIDI to audio
        audio_data = MIDIToAudio.convert(midi_data, sample_rate, self.sf2_path)
        
        # Save audio
        MIDIToAudio.save(audio_data, output_path, sample_rate)
    
    def wav_to_all_formats(
        self,
        wav_path: str,
        output_dir: str,
        base_name: Optional[str] = None,
        title: str = "Music Transcription",
        composer: str = "AI Transcription System"
    ) -> Dict[str, str]:
        """
        Convert WAV to all output formats.
        
        Args:
            wav_path (str): Path to WAV file
            output_dir (str): Directory to save output files
            base_name (str, optional): Base name for output files
            title (str): Title of the sheet music
            composer (str): Composer name
            
        Returns:
            Dict[str, str]: Dictionary of output file paths
        """
        if self.transcriber is None:
            raise ValueError("Transcriber is required for WAV to MIDI conversion")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine base name
        if base_name is None:
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
        
        # Transcribe WAV to MIDI
        result = self.transcriber.transcribe_file(wav_path)
        midi_data = result['midi_data']
        
        # Save MIDI
        midi_path = os.path.join(output_dir, f"{base_name}.mid")
        midi_data.write(midi_path)
        
        # Convert to other formats
        output_paths = {
            'midi': midi_path
        }
        
        try:
            # MusicXML
            musicxml_path = os.path.join(output_dir, f"{base_name}.musicxml")
            self.midi_to_musicxml(midi_data, musicxml_path)
            output_paths['musicxml'] = musicxml_path
        except Exception as e:
            print(f"Error converting to MusicXML: {e}")
        
        try:
            # PDF
            pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
            self.midi_to_pdf(midi_data, pdf_path, title, composer)
            output_paths['pdf'] = pdf_path
        except Exception as e:
            print(f"Error converting to PDF: {e}")
        
        try:
            # LilyPond
            ly_path = os.path.join(output_dir, f"{base_name}.ly")
            self.midi_to_lilypond(midi_data, ly_path, title, composer)
            output_paths['lilypond'] = ly_path
        except Exception as e:
            print(f"Error converting to LilyPond: {e}")
        
        try:
            # MuseScore
            mscz_path = os.path.join(output_dir, f"{base_name}.mscz")
            self.midi_to_musescore(midi_data, mscz_path)
            output_paths['musescore'] = mscz_path
        except Exception as e:
            print(f"Error converting to MuseScore: {e}")
        
        try:
            # Piano roll
            piano_roll_path = os.path.join(output_dir, f"{base_name}_piano_roll.png")
            self.midi_to_piano_roll(midi_data, piano_roll_path)
            output_paths['piano_roll'] = piano_roll_path
        except Exception as e:
            print(f"Error creating piano roll: {e}")
        
        try:
            # Audio
            audio_path = os.path.join(output_dir, f"{base_name}_synthesized.wav")
            self.midi_to_audio(midi_data, audio_path)
            output_paths['audio'] = audio_path
        except Exception as e:
            print(f"Error converting to audio: {e}")
        
        return output_paths


if __name__ == "__main__":
    # Example usage
    import argparse
    from inference.transcriber import MusicTranscriber, load_model
    
    parser = argparse.ArgumentParser(description="Convert between music formats")
    parser.add_argument("--input", type=str, required=True, help="Path to input file (MIDI or WAV)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output files")
    parser.add_argument("--format", type=str, default="all", help="Output format (midi, musicxml, pdf, lilypond, musescore, piano_roll, audio, all)")
    parser.add_argument("--title", type=str, default="Music Transcription", help="Title of the sheet music")
    parser.add_argument("--composer", type=str, default="AI Transcription System", help="Composer name")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (required for WAV input)")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate for audio processing")
    parser.add_argument("--sf2_path", type=str, help="Path to SoundFont file for audio synthesis")
    
    args = parser.parse_args()
    
    # Determine input file type
    input_ext = os.path.splitext(args.input)[1].lower()
    
    # Create format converter
    if input_ext == '.wav' or input_ext == '.mp3':
        # Load model for transcription
        if args.checkpoint is None:
            raise ValueError("Model checkpoint is required for WAV/MP3 input")
        
        model = load_model(args.checkpoint)
        transcriber = MusicTranscriber(model=model, sample_rate=args.sample_rate)
        
        converter = FormatConverter(transcriber=transcriber, sample_rate=args.sample_rate, sf2_path=args.sf2_path)
        
        # Convert WAV to all formats
        output_paths = converter.wav_to_all_formats(
            args.input,
            args.output_dir,
            title=args.title,
            composer=args.composer
        )
        
        print("Conversion completed!")
        for format_name, path in output_paths.items():
            print(f"{format_name}: {path}")
    
    elif input_ext == '.mid' or input_ext == '.midi':
        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(args.input)
        
        converter = FormatConverter(sample_rate=args.sample_rate, sf2_path=args.sf2_path)
        
        # Determine base name
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        
        # Convert to specified format
        if args.format == 'all' or args.format == 'musicxml':
            musicxml_path = os.path.join(args.output_dir, f"{base_name}.musicxml")
            converter.midi_to_musicxml(midi_data, musicxml_path)
            print(f"MusicXML: {musicxml_path}")
        
        if args.format == 'all' or args.format == 'pdf':
            pdf_path = os.path.join(args.output_dir, f"{base_name}.pdf")
            converter.midi_to_pdf(midi_data, pdf_path, args.title, args.composer)
            print(f"PDF: {pdf_path}")
        
        if args.format == 'all' or args.format == 'lilypond':
            ly_path = os.path.join(args.output_dir, f"{base_name}.ly")
            converter.midi_to_lilypond(midi_data, ly_path, args.title, args.composer)
            print(f"LilyPond: {ly_path}")
        
        if args.format == 'all' or args.format == 'musescore':
            mscz_path = os.path.join(args.output_dir, f"{base_name}.mscz")
            try:
                converter.midi_to_musescore(midi_data, mscz_path)
                print(f"MuseScore: {mscz_path}")
            except ValueError as e:
                print(f"MuseScore conversion failed: {e}")
        
        if args.format == 'all' or args.format == 'piano_roll':
            piano_roll_path = os.path.join(args.output_dir, f"{base_name}_piano_roll.png")
            converter.midi_to_piano_roll(midi_data, piano_roll_path)
            print(f"Piano Roll: {piano_roll_path}")
        
        if args.format == 'all' or args.format == 'audio':
            audio_path = os.path.join(args.output_dir, f"{base_name}_synthesized.wav")
            converter.midi_to_audio(midi_data, audio_path)
            print(f"Audio: {audio_path}")
        
        print("Conversion completed!")
    
    else:
        print(f"Unsupported input file format: {input_ext}")
        print("Supported formats: .mid, .midi, .wav, .mp3")
