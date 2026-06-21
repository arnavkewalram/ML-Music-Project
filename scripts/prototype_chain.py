"""
Prototype of the Demucs -> basic-pitch -> music21 chain.

    full song --[Demucs]--> isolated instrument stem --[basic-pitch]--> MIDI
              --[music21]--> sheet music (MusicXML)

Demonstrates the realistic "song -> sheet music for one instrument" pipeline
using only pretrained, off-the-shelf models (no training). Defaults to a
Creative-Commons librosa example clip if no input file is given.

Usage:
    python scripts/prototype_chain.py [audio_file] [--stem piano]
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output", "chain")


def resolve_input(path):
    if path:
        return path
    import librosa
    print("No input given; using CC-licensed librosa example 'brahms'.")
    return librosa.example("brahms")


def separate(audio_path, target_stem):
    """Run Demucs and return (stem_wav_path, sample_rate, available_stems)."""
    import numpy as np
    import librosa
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    # 6-source model so we can target guitar / piano specifically
    print("[1] Loading Demucs (htdemucs_6s) and separating...")
    model = get_model("htdemucs_6s")
    model.eval()
    sr = model.samplerate

    # Load via librosa (avoids the ffmpeg dependency demucs.audio needs)
    audio, _ = librosa.load(audio_path, sr=sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])          # mono -> stereo
    elif audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    wav = torch.from_numpy(audio).float()
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    with torch.no_grad():
        sources = apply_model(model, wav[None], device="cpu", progress=True)[0]
    sources = sources * ref.std() + ref.mean()

    stems = {name: sources[i] for i, name in enumerate(model.sources)}
    available = list(stems.keys())
    print(f"    stems produced: {available}")

    if target_stem not in stems:
        raise ValueError(f"stem '{target_stem}' not in {available}")

    os.makedirs(OUT_DIR, exist_ok=True)
    stem_path = os.path.join(OUT_DIR, f"stem_{target_stem}.wav")
    wav_out = stems[target_stem].cpu().numpy().T  # (samples, channels)
    sf.write(stem_path, wav_out, sr)
    print(f"    saved isolated '{target_stem}' stem -> {stem_path}")
    return stem_path, sr, available


def transcribe(stem_path):
    """Run basic-pitch on the stem; return a pretty_midi object + note count."""
    from basic_pitch.inference import predict

    print("[2] Transcribing stem with basic-pitch...")
    _, midi_data, note_events = predict(stem_path)
    n_notes = sum(len(inst.notes) for inst in midi_data.instruments)
    print(f"    basic-pitch found {n_notes} notes")
    midi_path = os.path.join(OUT_DIR, "transcription.mid")
    midi_data.write(midi_path)
    print(f"    saved MIDI -> {midi_path}")
    return midi_data, n_notes


def notate(midi_data):
    """Convert the transcription MIDI to MusicXML sheet music via the repo converter."""
    from inference.format_converters import FormatConverter

    print("[3] Converting MIDI -> sheet music (MusicXML)...")
    xml_path = os.path.join(OUT_DIR, "sheet_music.musicxml")
    converter = FormatConverter()
    converter.midi_to_musicxml(midi_data, xml_path)
    print(f"    saved sheet music -> {xml_path}")
    return xml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", default=None, help="input song (defaults to a librosa example)")
    ap.add_argument("--stem", default="piano",
                    help="instrument stem to transcribe: drums|bass|vocals|guitar|piano|other")
    args = ap.parse_args()

    print("=" * 60)
    print("DEMUCS -> BASIC-PITCH -> MUSIC21 PROTOTYPE")
    print("=" * 60)

    audio_path = resolve_input(args.audio)
    print(f"Input song: {audio_path}")
    print(f"Target instrument stem: {args.stem}\n")

    stem_path, sr, available = separate(audio_path, args.stem)
    midi_data, n_notes = transcribe(stem_path)
    xml_path = notate(midi_data)

    print("\n" + "=" * 60)
    print("DONE. Full chain ran on a real song.")
    print(f"  isolated stem : {stem_path}")
    print(f"  transcription : {n_notes} notes")
    print(f"  sheet music   : {xml_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
