"""
Integration test for the training pipeline on a tiny, MAESTRO-formatted dataset.

This does NOT need the 101GB MAESTRO download. It synthesizes a couple of
audio+MIDI pairs (sine-tone scales) laid out exactly like MAESTRO, then runs the
*real* code paths:

    preprocess_maestro() -> PreprocessedMAESTRODataset -> create_datasets() -> Trainer

The point is to catch bugs in preprocessing, dataset collation, and the training
loop before committing to a full download + cloud GPU run. It also confirms the
model can take a real optimizer step (loss is finite and weights update).
"""

import os
import sys
import json
import shutil
import tempfile
import numpy as np
import pretty_midi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from preprocessing.dataset_preprocessor import preprocess_maestro, create_datasets
from model.architecture import MusicTranscriptionModel
from model.trainer import Trainer, create_optimizer

SAMPLE_RATE = 22050
SCALE = [60, 62, 64, 65, 67, 69, 71, 72]
NOTE_DUR = 0.5


def midi_to_freq(p):
    return 440.0 * (2.0 ** ((p - 69) / 12.0))


def make_pair(wav_path, midi_path):
    """Write a synthesized scale as a .wav and a matching .mid."""
    import soundfile as sf

    audio = np.array([], dtype=np.float32)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    t = np.linspace(0, NOTE_DUR, int(SAMPLE_RATE * NOTE_DUR), endpoint=False)
    env = np.ones_like(t)
    a, r = int(0.01 * SAMPLE_RATE), int(0.05 * SAMPLE_RATE)
    env[:a] = np.linspace(0, 1, a)
    env[-r:] = np.linspace(1, 0, r)

    for i, pitch in enumerate(SCALE):
        f0 = midi_to_freq(pitch)
        tone = (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)) * env
        tone = (tone / np.max(np.abs(tone))).astype(np.float32)
        audio = np.concatenate([audio, tone])
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch,
                                            start=i * NOTE_DUR, end=(i + 1) * NOTE_DUR))
    pm.instruments.append(inst)
    sf.write(wav_path, audio, SAMPLE_RATE)
    pm.write(midi_path)
    return len(audio) / SAMPLE_RATE


def build_fake_maestro(root):
    """Create a minimal MAESTRO-format directory with train/validation/test pairs."""
    rows = []
    for split in ['train', 'validation', 'test']:
        audio_fn = f"{split}/piece.wav"
        midi_fn = f"{split}/piece.midi"
        os.makedirs(os.path.join(root, split), exist_ok=True)
        dur = make_pair(os.path.join(root, audio_fn), os.path.join(root, midi_fn))
        rows.append({
            'canonical_composer': 'Test Composer',
            'canonical_title': f'Test Scale ({split})',
            'split': split,
            'year': 2025,
            'midi_filename': midi_fn,
            'audio_filename': audio_fn,
            'duration': dur,
        })
    with open(os.path.join(root, 'maestro-v3.0.0.json'), 'w') as f:
        json.dump(rows, f)


def main():
    tmp = tempfile.mkdtemp(prefix='maestro_mini_')
    raw = os.path.join(tmp, 'raw')
    pre = os.path.join(tmp, 'preprocessed')
    os.makedirs(raw, exist_ok=True)
    try:
        print("=" * 60)
        print("TRAINING PIPELINE INTEGRATION TEST")
        print("=" * 60)

        print("\n[1] Building tiny MAESTRO-format dataset...")
        build_fake_maestro(raw)

        print("[2] Running real preprocess_maestro()...")
        stats = preprocess_maestro(
            dataset_dir=raw, output_dir=pre,
            segment_duration=2.0, overlap=0.5,
        )
        print(f"    processed={stats['processed_items']} "
              f"skipped={stats['skipped_items']} segments={stats['total_segments']}")
        assert stats['processed_items'] == 3, "preprocessing dropped items"
        assert stats['skipped_items'] == 0, "preprocessing skipped items (errors)"

        print("[3] Building DataLoaders from preprocessed data...")
        loaders = create_datasets(preprocessed_dir=pre, batch_size=2, num_workers=0)
        print(f"    train segments: {len(loaders['train'].dataset)}")
        assert len(loaders['train'].dataset) > 0, "no training segments found"

        print("[4] Running real training steps (3 epochs on tiny data)...")
        model = MusicTranscriptionModel()
        optimizer = create_optimizer(model, learning_rate=1e-3)
        trainer = Trainer(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders['validation'],
            optimizer=optimizer,
            device=torch.device('cpu'),
            checkpoint_dir=os.path.join(tmp, 'ckpt'),
            log_dir=os.path.join(tmp, 'logs'),
        )

        w_before = next(model.parameters()).clone()
        losses = []
        for epoch in range(3):
            tr = trainer.train_epoch()
            losses.append(tr['total'])
            print(f"    epoch {epoch + 1}: train_loss={tr['total']:.4f}")
        w_after = next(model.parameters())

        assert all(np.isfinite(l) for l in losses), "non-finite loss"
        assert not torch.equal(w_before, w_after), "weights did not update"

        print("\n" + "=" * 60)
        print("PASS: full training pipeline runs on real data.")
        print(f"  - preprocessing: OK ({stats['total_segments']} segments)")
        print(f"  - dataloader/collation: OK")
        print(f"  - optimizer step updates weights: OK")
        print(f"  - loss finite across epochs: {[f'{l:.3f}' for l in losses]}")
        print("=" * 60)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
