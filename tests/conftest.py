"""Shared fixtures.

Design rule for this suite: nothing outside the `slow` marker may load Demucs,
basic-pitch or the ByteDance piano model. Those are hundreds of megabytes and
tens of seconds each, which would make the suite useless as a feedback loop. The
fast tests exercise our own logic — quantization, notation, routing, the HTTP
contract — against synthetic audio, and stub the heavy models out.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

SAMPLE_RATE = 22050


# --------------------------------------------------------------- audio fixtures
def write_tone(path: str, midi_pitch: float = 69.0, duration: float = 1.0,
               sample_rate: int = SAMPLE_RATE, vibrato_semitones: float = 0.0,
               vibrato_hz: float = 5.5) -> str:
    """Render a harmonic tone at a MIDI pitch and write it to `path`.

    Two harmonics rather than a bare sine: pYIN and the onset detectors both key
    off spectral structure, and a pure sine is unrepresentatively easy.
    """
    import librosa

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    pitch_track = midi_pitch + vibrato_semitones * np.sin(2 * np.pi * vibrato_hz * t)
    f0 = librosa.midi_to_hz(pitch_track)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    y = 0.6 * np.sin(phase) + 0.2 * np.sin(2 * phase)

    # Short fade in/out so the file has clean onsets and no click artefacts.
    fade = int(0.01 * sample_rate)
    if len(y) > 2 * fade:
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)

    sf.write(path, y.astype(np.float32), sample_rate)
    return path


@pytest.fixture
def tone_wav(tmp_path):
    """A 1s A4 tone — the default 'this is valid audio' input."""
    return write_tone(str(tmp_path / "tone.wav"), midi_pitch=69.0, duration=1.0)


@pytest.fixture
def tone_factory(tmp_path):
    """Build extra tones inside a test: `tone_factory(midi_pitch=40, ...)`."""
    counter = {"n": 0}

    def _make(**kwargs):
        counter["n"] += 1
        path = str(tmp_path / f"tone_{counter['n']}.wav")
        return write_tone(path, **kwargs)

    return _make


@pytest.fixture
def not_audio(tmp_path):
    """A file with an audio extension that is not decodable audio."""
    path = tmp_path / "broken.wav"
    path.write_bytes(b"this is definitely not a RIFF header")
    return str(path)


# ---------------------------------------------------------------- midi fixtures
def make_midi(notes, initial_tempo: float = 120.0, is_drum: bool = False,
              program: int = 0) -> pretty_midi.PrettyMIDI:
    """Build a PrettyMIDI from `(pitch, start, end)` triples."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
    inst.notes = [
        pretty_midi.Note(velocity=90, pitch=p, start=s, end=e)
        for p, s, e in notes
    ]
    pm.instruments.append(inst)
    return pm


@pytest.fixture
def midi_factory():
    return make_midi


# ------------------------------------------------------------------ API fixture
@pytest.fixture
def client(monkeypatch, tmp_path):
    """A TestClient with the heavy models stubbed out and an isolated runs dir.

    `pipeline.run` is replaced with a fake that writes the same artefacts the
    real one does, so the HTTP contract is tested without a 40-second model load.
    """
    from fastapi.testclient import TestClient

    from webapp import pipeline, server

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(server, "RUNS_DIR", str(runs_dir))
    monkeypatch.setattr(pipeline, "warmup", lambda: None)

    calls = []

    def fake_run(audio_path, stem, work_dir):
        calls.append({"audio_path": audio_path, "stem": stem, "work_dir": work_dir})
        for name in (f"stem_{stem}.wav", "transcription.mid", "sheet_music.musicxml"):
            with open(os.path.join(work_dir, name), "w") as fh:
                fh.write("stub")
        return {
            "stem_path": os.path.join(work_dir, f"stem_{stem}.wav"),
            "midi_path": os.path.join(work_dir, "transcription.mid"),
            "musicxml_path": os.path.join(work_dir, "sheet_music.musicxml"),
            "musicxml": "<score-partwise/>",
            "n_notes": 42,
            "duration": 12.34,
            "method": "stub transcriber",
            "tempo": 100.0,
            "stem_energy_ratio": 0.5,
        }

    monkeypatch.setattr(pipeline, "run", fake_run)

    with TestClient(server.app) as c:
        c.pipeline_calls = calls
        yield c
