"""Tests for per-instrument transcription (webapp/transcribers.py).

The pretrained models themselves are not under test here — their accuracy is
their maintainers' problem. What is under test is our routing, our pYIN note
segmentation, and our drum classification.
"""

import warnings

warnings.filterwarnings("ignore")

import pytest

from webapp import pipeline, transcribers


class TestRouter:
    def test_every_demucs_stem_has_a_transcriber(self):
        assert set(pipeline.VALID_STEMS) == set(transcribers.ROUTER)

    def test_an_unknown_stem_routes_to_the_polyphonic_model(self, monkeypatch):
        called = {}
        monkeypatch.setitem(
            transcribers.ROUTER, "piano", lambda p: (_ for _ in ()).throw(AssertionError)
        )
        monkeypatch.setattr(
            transcribers, "transcribe_polyphonic",
            lambda p: (called.setdefault("path", p), "poly"),
        )

        transcribers.transcribe("theremin", "/nonexistent.wav")

        assert called["path"] == "/nonexistent.wav"

    def test_a_failing_specialist_falls_back_to_the_polyphonic_model(self, monkeypatch):
        """One broken specialist must not turn into a failed user request."""
        def broken(path):
            raise RuntimeError("specialist exploded")

        monkeypatch.setitem(transcribers.ROUTER, "bass", broken)
        monkeypatch.setattr(
            transcribers, "transcribe_polyphonic", lambda p: ("fallback-midi", "poly")
        )

        midi, method = transcribers.transcribe("bass", "/nonexistent.wav")

        assert (midi, method) == ("fallback-midi", "poly")

    def test_a_failing_polyphonic_model_propagates(self, monkeypatch):
        """There is nothing left to fall back to, so the caller must be told."""
        def broken(path):
            raise RuntimeError("basic-pitch exploded")

        monkeypatch.setattr(transcribers, "transcribe_polyphonic", broken)
        monkeypatch.setitem(transcribers.ROUTER, "other", broken)

        with pytest.raises(RuntimeError, match="basic-pitch exploded"):
            transcribers.transcribe("other", "/nonexistent.wav")


class TestMonophonicPitchTracking:
    def test_a_steady_tone_is_transcribed_at_the_right_pitch(self, tone_factory):
        path = tone_factory(midi_pitch=57.0, duration=1.5)  # A3

        pm = transcribers._f0_to_midi(path, fmin=82.0, fmax=1000.0, program=53)

        pitches = [n.pitch for n in pm.instruments[0].notes]
        assert pitches, "expected at least one note"
        assert max(set(pitches), key=pitches.count) == 57

    def test_the_note_covers_the_length_of_the_tone(self, tone_factory):
        path = tone_factory(midi_pitch=57.0, duration=1.5)

        pm = transcribers._f0_to_midi(path, fmin=82.0, fmax=1000.0, program=53)

        notated = sum(n.end - n.start for n in pm.instruments[0].notes)
        assert notated > 0.7 * 1.5

    def test_silence_produces_no_notes(self, tmp_path):
        import numpy as np
        import soundfile as sf

        path = tmp_path / "silence.wav"
        sf.write(str(path), np.zeros(22050, dtype="float32"), 22050)

        pm = transcribers._f0_to_midi(str(path), fmin=82.0, fmax=1000.0, program=53)

        assert pm.instruments[0].notes == []

    def test_the_program_number_is_applied(self, tone_factory):
        path = tone_factory(midi_pitch=57.0, duration=0.6)
        pm = transcribers._f0_to_midi(path, fmin=82.0, fmax=1000.0, program=33)
        assert pm.instruments[0].program == 33

    def test_bass_and_vocals_use_their_own_registers(self, tone_factory):
        """A bass note below the vocal fmin must still be found by the bass path."""
        low = tone_factory(midi_pitch=33.0, duration=1.2)  # A1, below vocals' 82 Hz floor

        bass_pm, bass_method = transcribers.transcribe_bass(low)
        vocal_pm, _ = transcribers.transcribe_vocals(low)

        assert "bass" in bass_method
        assert bass_pm.instruments[0].notes, "bass register should find a low A"
        bass_pitches = [n.pitch for n in bass_pm.instruments[0].notes]
        assert abs(max(set(bass_pitches), key=bass_pitches.count) - 33) <= 1
        # The vocal range starts at E2, so it cannot honestly report A1.
        assert all(n.pitch >= 38 for n in vocal_pm.instruments[0].notes)


class TestDrums:
    def test_output_is_flagged_as_a_percussion_track(self, tone_factory):
        path = tone_factory(midi_pitch=60.0, duration=1.0)

        pm, method = transcribers.transcribe_drums(path)

        assert pm.instruments[0].is_drum is True
        assert "onset" in method

    def test_hits_are_mapped_into_the_general_midi_drum_set(self, tmp_path):
        import numpy as np
        import soundfile as sf

        sr = 22050
        y = np.zeros(int(sr * 2.0), dtype="float32")
        # Alternating low thud (kick) and bright noise burst (hi-hat).
        for i, t in enumerate(np.arange(0.1, 1.9, 0.25)):
            start = int(t * sr)
            n = int(0.05 * sr)
            env = np.exp(-np.linspace(0, 8, n))
            if i % 2 == 0:
                hit = np.sin(2 * np.pi * 60 * np.linspace(0, 0.05, n)) * env
            else:
                hit = np.random.RandomState(i).randn(n) * env * 0.5
            y[start:start + n] += hit.astype("float32")
        path = tmp_path / "drums.wav"
        sf.write(str(path), y, sr)

        pm, _ = transcribers.transcribe_drums(str(path))

        pitches = {n.pitch for n in pm.instruments[0].notes}
        assert pitches, "expected onsets to be detected"
        assert pitches <= {transcribers.DRUM_KICK, transcribers.DRUM_SNARE,
                           transcribers.DRUM_HIHAT}

    def test_silence_produces_no_hits(self, tmp_path):
        import numpy as np
        import soundfile as sf

        path = tmp_path / "silence.wav"
        sf.write(str(path), np.zeros(22050, dtype="float32"), 22050)

        pm, _ = transcribers.transcribe_drums(str(path))

        assert pm.instruments[0].notes == []
