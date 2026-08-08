"""Tests for the rhythm-quantization stage (webapp/pipeline.py)."""

import warnings

warnings.filterwarnings("ignore")

import librosa
import pytest

from webapp import pipeline


@pytest.fixture
def fixed_tempo(monkeypatch):
    """Pin librosa's beat tracker so grid maths is deterministic.

    Beat tracking on a 1-second synthetic tone is essentially arbitrary; these
    tests are about the snapping, not about the estimator.
    """

    def _set(bpm):
        monkeypatch.setattr(librosa.beat, "beat_track", lambda **kw: (bpm, None))
        return bpm

    return _set


def test_snaps_onsets_to_the_sixteenth_grid(tone_wav, midi_factory, fixed_tempo):
    fixed_tempo(120.0)  # grid = (60/120)/4 = 0.125s
    # Deliberately ragged times, each within half a grid step of a grid line.
    midi = midi_factory([(60, 0.13, 0.38), (62, 0.51, 0.74)])

    out, tempo = pipeline.quantize_to_grid(midi, tone_wav)

    assert tempo == 120.0
    times = [(n.start, n.end) for n in out.instruments[0].notes]
    assert times == [(0.125, 0.375), (0.5, 0.75)]


def test_tempo_is_reported_at_one_decimal(tone_wav, midi_factory, fixed_tempo):
    fixed_tempo(123.456)
    _, tempo = pipeline.quantize_to_grid(midi_factory([(60, 0.0, 1.0)]), tone_wav)
    assert tempo == 123.5


@pytest.mark.parametrize("bpm", [0.0, 12.0, 900.0])
def test_absurd_tempo_estimates_fall_back_to_120(tone_wav, midi_factory, fixed_tempo, bpm):
    fixed_tempo(bpm)
    _, tempo = pipeline.quantize_to_grid(midi_factory([(60, 0.0, 1.0)]), tone_wav)
    assert tempo == 120.0


@pytest.mark.parametrize("bpm", [30.0, 39.8, 300.0])
def test_slow_but_real_tempi_are_kept(tone_wav, midi_factory, fixed_tempo, bpm):
    """A genuine 40 BPM largo reads as ~39.8; it must not be clamped to 120."""
    fixed_tempo(bpm)
    _, tempo = pipeline.quantize_to_grid(midi_factory([(60, 0.0, 1.0)]), tone_wav)
    assert tempo == round(bpm, 1)


def test_a_note_shorter_than_the_grid_survives_as_one_grid_step(
    tone_wav, midi_factory, fixed_tempo
):
    """Rounding start and end independently can collapse a note to zero length."""
    fixed_tempo(120.0)
    midi = midi_factory([(60, 0.50, 0.52)])

    out, _ = pipeline.quantize_to_grid(midi, tone_wav)

    note = out.instruments[0].notes[0]
    assert note.end > note.start
    assert note.end - note.start == pytest.approx(0.125)


def test_beat_tracking_failure_falls_back_instead_of_raising(
    tone_wav, midi_factory, monkeypatch
):
    def boom(**kwargs):
        raise RuntimeError("beat tracker exploded")

    monkeypatch.setattr(librosa.beat, "beat_track", boom)

    out, tempo = pipeline.quantize_to_grid(midi_factory([(60, 0.0, 0.5)]), tone_wav)

    assert tempo == 120.0
    assert len(out.instruments[0].notes) == 1


def test_drum_hits_are_never_merged(tone_wav, midi_factory, fixed_tempo):
    """Repeated hi-hat hits are the whole point of a drum part."""
    fixed_tempo(120.0)
    hits = [(42, t, t + 0.05) for t in (0.0, 0.125, 0.25, 0.375)]
    midi = midi_factory(hits, is_drum=True)

    out, _ = pipeline.quantize_to_grid(midi, tone_wav)

    assert len(out.instruments[0].notes) == 4


def test_instrument_metadata_survives_quantization(tone_wav, midi_factory, fixed_tempo):
    fixed_tempo(120.0)
    midi = midi_factory([(36, 0.0, 0.1)], is_drum=True, program=0)
    midi.instruments[0].name = "Drums"

    out, _ = pipeline.quantize_to_grid(midi, tone_wav)

    assert out.instruments[0].is_drum is True
    assert out.instruments[0].name == "Drums"


class TestMergeSamePitch:
    def test_overlapping_fragments_of_one_note_become_one_note(self):
        """A sustained note split into slivers is the case this exists to fix."""
        import pretty_midi

        slivers = [
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.30),
            pretty_midi.Note(velocity=95, pitch=60, start=0.25, end=0.55),
            pretty_midi.Note(velocity=70, pitch=60, start=0.50, end=0.80),
        ]

        merged = pipeline._merge_same_pitch(slivers)

        assert len(merged) == 1
        assert (merged[0].start, merged[0].end) == (0.0, 0.80)
        assert merged[0].velocity == 95  # loudest fragment wins

    def test_different_pitches_are_never_merged(self):
        import pretty_midi

        notes = [
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5),
            pretty_midi.Note(velocity=80, pitch=64, start=0.0, end=0.5),
            pretty_midi.Note(velocity=80, pitch=67, start=0.0, end=0.5),
        ]

        merged = pipeline._merge_same_pitch(notes)

        assert sorted(n.pitch for n in merged) == [60, 64, 67]

    def test_output_is_sorted_by_start_then_pitch(self):
        import pretty_midi

        notes = [
            pretty_midi.Note(velocity=80, pitch=67, start=1.0, end=1.5),
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5),
            pretty_midi.Note(velocity=80, pitch=64, start=0.0, end=0.5),
        ]

        merged = pipeline._merge_same_pitch(notes)

        assert [(n.start, n.pitch) for n in merged] == [(0.0, 60), (0.0, 64), (1.0, 67)]

    def test_empty_input_is_handled(self):
        assert pipeline._merge_same_pitch([]) == []
