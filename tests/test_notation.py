"""Tests for MIDI -> MusicXML engraving (webapp/pipeline.py)."""

import warnings

warnings.filterwarnings("ignore")

import pytest
from music21 import converter

from webapp import pipeline


def parse(xml_path):
    return converter.parse(xml_path)


def midi_pitches(stream):
    """Every sounding MIDI pitch in a stream, chords included."""
    return sorted(p.midi for n in stream.recurse().notes for p in n.pitches)


def test_notate_writes_parseable_musicxml(tmp_path, midi_factory):
    midi = midi_factory([(60, 0.0, 0.5), (62, 0.5, 1.0), (64, 1.0, 1.5)])
    out = tmp_path / "score.musicxml"

    xml = pipeline.notate(midi, str(out))

    assert out.exists()
    assert xml.startswith("<?xml")
    assert len(list(parse(out).recurse().notes)) == 3


def test_notate_returns_the_same_text_it_wrote(tmp_path, midi_factory):
    out = tmp_path / "score.musicxml"
    xml = pipeline.notate(midi_factory([(60, 0.0, 1.0)]), str(out))
    assert xml == out.read_text(encoding="utf-8")


def test_an_empty_transcription_does_not_crash_the_engraver(tmp_path):
    """An absent instrument yields no notes; that must still produce a file."""
    import pretty_midi

    empty = pretty_midi.PrettyMIDI(initial_tempo=120)
    empty.instruments.append(pretty_midi.Instrument(program=0))
    out = tmp_path / "empty.musicxml"

    xml = pipeline.notate(empty, str(out))

    assert out.exists()
    assert "score-partwise" in xml


def test_notate_leaves_no_temporary_midi_behind(tmp_path, midi_factory):
    import glob
    import tempfile

    before = set(glob.glob(f"{tempfile.gettempdir()}/*.mid"))
    pipeline.notate(midi_factory([(60, 0.0, 1.0)]), str(tmp_path / "s.musicxml"))
    after = set(glob.glob(f"{tempfile.gettempdir()}/*.mid"))

    assert after == before


class TestGrandStaff:
    def test_piano_is_engraved_on_two_staves(self, tmp_path, midi_factory):
        midi = midi_factory([(72, 0.0, 1.0), (48, 0.0, 1.0)])
        out = tmp_path / "piano.musicxml"

        pipeline.notate(midi, str(out), stem="piano")

        assert len(parse(out).parts) == 2

    def test_non_piano_stems_stay_on_one_staff(self, tmp_path, midi_factory):
        midi = midi_factory([(72, 0.0, 1.0), (48, 0.0, 1.0)])
        out = tmp_path / "bass.musicxml"

        pipeline.notate(midi, str(out), stem="bass")

        assert len(parse(out).parts) == 1

    def test_pitches_are_split_at_middle_c(self, tmp_path, midi_factory):
        """High notes belong on the treble staff, low notes on the bass staff."""
        midi = midi_factory([(72, 0.0, 1.0), (48, 0.0, 1.0)])
        out = tmp_path / "split.musicxml"

        pipeline.notate(midi, str(out), stem="piano")

        by_staff = [midi_pitches(p) for p in parse(out).parts]
        assert [72] in by_staff
        assert [48] in by_staff

    def test_chords_are_split_across_both_staves(self, tmp_path, midi_factory):
        """A two-hand chord must not be dumped whole onto one staff."""
        midi = midi_factory([(76, 0.0, 1.0), (72, 0.0, 1.0), (48, 0.0, 1.0), (43, 0.0, 1.0)])
        out = tmp_path / "chord.musicxml"

        pipeline.notate(midi, str(out), stem="piano")

        pitch_sets = [set(midi_pitches(p)) for p in parse(out).parts]
        assert {72, 76} in pitch_sets
        assert {43, 48} in pitch_sets

    def test_notes_keep_their_absolute_position_in_time(self, tmp_path, midi_factory):
        """Regression: a flatten/recurse mix-up once collapsed every note to bar 1."""
        midi = midi_factory([(72, 0.0, 1.0), (74, 2.0, 3.0), (76, 4.0, 5.0)])
        out = tmp_path / "timing.musicxml"

        pipeline.notate(midi, str(out), stem="piano")

        # Offsets are quarterLengths: at 120 BPM a quarter note is 0.5s, so the
        # notes at 0s / 2s / 4s belong at quarter offsets 0 / 4 / 8.
        score = parse(out)
        offsets = sorted(n.getOffsetInHierarchy(score) for n in score.recurse().notes)
        assert offsets == [0.0, 4.0, 8.0]


class TestStemEnergyRatio:
    def test_a_silent_stem_scores_near_zero(self, tmp_path, tone_wav):
        import numpy as np
        import soundfile as sf

        silent = tmp_path / "silent.wav"
        sf.write(str(silent), np.zeros(22050, dtype="float32"), 22050)

        assert pipeline._stem_energy_ratio(tone_wav, str(silent)) < 0.01

    def test_a_stem_identical_to_the_mix_scores_one(self, tone_wav):
        assert pipeline._stem_energy_ratio(tone_wav, tone_wav) == pytest.approx(1.0, abs=1e-3)

    def test_unreadable_files_do_not_raise(self, not_audio, tone_wav):
        """On error we must not flag a real instrument as absent."""
        assert pipeline._stem_energy_ratio(tone_wav, not_audio) == 1.0
