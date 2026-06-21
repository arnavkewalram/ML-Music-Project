"""
End-to-end pipeline test for the music transcription system.

Generates a synthetic audio clip with known ground-truth notes (a C-major
scale of harmonic tones), runs it through the full transcriber, exports the
result to sheet music (MusicXML), and scores the transcription against the
known ground truth with mir_eval.

This validates the *plumbing* of the whole loop:
    audio -> features -> model -> notes -> MIDI -> sheet music -> metrics

Note: with an untrained (randomly initialized) model the accuracy numbers are
expected to be near zero. This script measures that the loop runs and produces
well-formed output + real metrics, not that the model is accurate.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.architecture import MusicTranscriptionModel
from model.evaluation import calculate_note_metrics, calculate_onset_metrics
from inference.transcriber import MusicTranscriber
from inference.format_converters import FormatConverter

SAMPLE_RATE = 22050
NOTE_DURATION = 0.5  # seconds per note
C_MAJOR_SCALE = [60, 62, 64, 65, 67, 69, 71, 72]  # MIDI pitches C4..C5


def midi_to_freq(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def synthesize_scale() -> tuple:
    """Render a harmonic-tone C-major scale and return (audio, ground_truth_notes)."""
    audio = np.array([], dtype=np.float32)
    notes = []
    t = np.linspace(0, NOTE_DURATION, int(SAMPLE_RATE * NOTE_DURATION), endpoint=False)

    # Simple attack/decay envelope so onsets are detectable
    env = np.ones_like(t)
    attack = int(0.01 * SAMPLE_RATE)
    release = int(0.05 * SAMPLE_RATE)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)

    for i, pitch in enumerate(C_MAJOR_SCALE):
        f0 = midi_to_freq(pitch)
        # Fundamental + two harmonics for a richer spectrum
        tone = (np.sin(2 * np.pi * f0 * t)
                + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
                + 0.25 * np.sin(2 * np.pi * 3 * f0 * t))
        tone = (tone / np.max(np.abs(tone))) * env
        onset = i * NOTE_DURATION
        notes.append({
            'pitch': pitch,
            'onset_time': onset,
            'offset_time': onset + NOTE_DURATION,
            'duration_time': NOTE_DURATION,
            'velocity': 100,
        })
        audio = np.concatenate([audio, tone.astype(np.float32)])

    return audio, notes


def main() -> int:
    # Deterministic run so results are reproducible
    torch.manual_seed(0)
    np.random.seed(0)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_output')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("END-TO-END PIPELINE TEST")
    print("=" * 60)

    # 1. Synthesize audio + ground truth
    audio, ref_notes = synthesize_scale()
    print(f"\n[1] Synthesized {len(audio) / SAMPLE_RATE:.1f}s of audio, "
          f"{len(ref_notes)} ground-truth notes (C-major scale).")

    # 2. Build model + transcriber (UNTRAINED weights)
    model = MusicTranscriptionModel()
    transcriber = MusicTranscriber(model=model, sample_rate=SAMPLE_RATE)
    print("[2] Built transcriber with randomly-initialized model.")

    # 3. Run transcription
    result = transcriber.transcribe_audio(audio, segment_duration=5.0, overlap=0.5)
    est_notes = result['notes']
    print(f"[3] Transcription produced {len(est_notes)} notes, "
          f"tempo={result['tempo']:.1f}, time_sig={result['time_signature']}.")

    # 4. Export to sheet music (MusicXML) + MIDI
    midi_path = os.path.join(out_dir, 'transcription.mid')
    transcriber.save_midi(result['midi_data'], midi_path)
    converter = FormatConverter(sample_rate=SAMPLE_RATE)
    xml_path = os.path.join(out_dir, 'transcription.musicxml')
    try:
        converter.midi_to_musicxml(result['midi_data'], xml_path)
        print(f"[4] Exported sheet music: {xml_path}")
    except Exception as e:
        print(f"[4] Sheet music export failed: {type(e).__name__}: {e}")

    # 5. Score against ground truth
    note_metrics = calculate_note_metrics(ref_notes, est_notes)
    ref_onsets = np.array([n['onset_time'] for n in ref_notes])
    est_onsets = np.array(sorted(n['onset_time'] for n in est_notes))
    onset_metrics = calculate_onset_metrics(ref_onsets, est_onsets)

    print("\n" + "=" * 60)
    print("RESULTS (vs. ground-truth sheet music)")
    print("=" * 60)
    print(f"  Note  precision: {note_metrics['precision']:.3f}")
    print(f"  Note  recall:    {note_metrics['recall']:.3f}")
    print(f"  Note  F1:        {note_metrics['f1_score']:.3f}")
    print(f"  Onset precision: {onset_metrics['precision']:.3f}")
    print(f"  Onset recall:    {onset_metrics['recall']:.3f}")
    print(f"  Onset F1:        {onset_metrics['f1_score']:.3f}")
    print("=" * 60)
    print("\nPipeline ran end-to-end. (Accuracy is expected to be ~0 until the "
          "model is trained.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
