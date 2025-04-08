"""
Main script for inference with the music transcription model.

This script provides a command-line interface for transcribing audio files
and converting them to various output formats.
"""

import os
import argparse
import torch

from model.architecture import MusicTranscriptionModel
from inference.transcriber import load_model, MusicTranscriber, RealTimeTranscriber
from inference.format_converters import FormatConverter


def main():
    parser = argparse.ArgumentParser(description="Music transcription inference")
    
    # Input options
    parser.add_argument("--input", type=str, help="Path to input audio file")
    parser.add_argument("--real_time", action="store_true", help="Use real-time transcription")
    parser.add_argument("--list_devices", action="store_true", help="List audio input devices")
    parser.add_argument("--device_index", type=int, help="Index of audio input device to use")
    
    # Model options
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate for audio processing")
    parser.add_argument("--onset_threshold", type=float, default=0.5, help="Threshold for onset detection")
    parser.add_argument("--pitch_threshold", type=float, default=0.5, help="Threshold for pitch detection")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output files")
    parser.add_argument("--format", type=str, default="all", 
                        help="Output format (midi, musicxml, pdf, lilypond, musescore, piano_roll, audio, all)")
    parser.add_argument("--title", type=str, default="Music Transcription", help="Title of the sheet music")
    parser.add_argument("--composer", type=str, default="AI Transcription System", help="Composer name")
    parser.add_argument("--visualize", action="store_true", help="Visualize transcription")
    parser.add_argument("--sf2_path", type=str, help="Path to SoundFont file for audio synthesis")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Create transcriber
    transcriber = MusicTranscriber(
        model=model,
        sample_rate=args.sample_rate,
        onset_threshold=args.onset_threshold,
        pitch_threshold=args.pitch_threshold
    )
    
    # Create format converter
    converter = FormatConverter(
        transcriber=transcriber,
        sample_rate=args.sample_rate,
        sf2_path=args.sf2_path
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
            if args.output_dir:
                midi_path = os.path.join(args.output_dir, "real_time_transcription.mid")
                rt_transcriber.save_transcription(midi_path)
                
                # Convert to other formats if requested
                if args.format != "midi":
                    # Load MIDI file
                    import pretty_midi
                    midi_data = pretty_midi.PrettyMIDI(midi_path)
                    
                    # Convert to specified format
                    if args.format == "all" or args.format == "musicxml":
                        musicxml_path = os.path.join(args.output_dir, "real_time_transcription.musicxml")
                        converter.midi_to_musicxml(midi_data, musicxml_path)
                    
                    if args.format == "all" or args.format == "pdf":
                        pdf_path = os.path.join(args.output_dir, "real_time_transcription.pdf")
                        converter.midi_to_pdf(midi_data, pdf_path, args.title, args.composer)
                    
                    if args.format == "all" or args.format == "lilypond":
                        ly_path = os.path.join(args.output_dir, "real_time_transcription.ly")
                        converter.midi_to_lilypond(midi_data, ly_path, args.title, args.composer)
                    
                    if args.format == "all" or args.format == "musescore":
                        try:
                            mscz_path = os.path.join(args.output_dir, "real_time_transcription.mscz")
                            converter.midi_to_musescore(midi_data, mscz_path)
                        except ValueError as e:
                            print(f"MuseScore conversion failed: {e}")
                    
                    if args.format == "all" or args.format == "piano_roll":
                        piano_roll_path = os.path.join(args.output_dir, "real_time_transcription_piano_roll.png")
                        converter.midi_to_piano_roll(midi_data, piano_roll_path)
                    
                    if args.format == "all" or args.format == "audio":
                        audio_path = os.path.join(args.output_dir, "real_time_transcription_synthesized.wav")
                        converter.midi_to_audio(midi_data, audio_path)
        
        except KeyboardInterrupt:
            # Stop real-time transcription
            rt_transcriber.stop()
    
    elif args.input:
        # Determine base name
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        
        # Convert to all formats
        output_paths = converter.wav_to_all_formats(
            args.input,
            args.output_dir,
            base_name=base_name,
            title=args.title,
            composer=args.composer
        )
        
        print("Transcription completed!")
        for format_name, path in output_paths.items():
            print(f"{format_name}: {path}")
        
        # Visualize if requested
        if args.visualize:
            # Transcribe audio file
            result = transcriber.transcribe_file(args.input)
            
            # Load audio for visualization
            audio_data, _ = transcriber.audio_loader.load_file(args.input)
            
            # Visualize transcription
            vis_output = os.path.join(args.output_dir, f"{base_name}_transcription.png")
            transcriber.visualize_transcription(result['notes'], audio_data, vis_output)
            print(f"Visualization saved to {vis_output}")
    
    else:
        print("Please provide an input audio file or use real-time transcription")


if __name__ == "__main__":
    main()
