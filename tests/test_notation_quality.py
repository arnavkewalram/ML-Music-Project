"""Tests for the musical accuracy of the notated output.

These cover things a musician would notice reading the score: repeated notes
turning into one long note, chords appearing in a single-line part, a page of
accidentals because no key was detected, and sustained notes vanishing.
"""

import re
import warnings

warnings.filterwarnings("ignore")

import librosa
import pretty_midi
import pytest
from music21 import converter

from webapp import pipeline, transcribers


@pytest.fixture
def at_120bpm(monkeypatch):
    """Pin the tempo so the grid is exactly 0.125s and the maths is checkable."""
    monkeypatch.setattr(librosa.beat, "beat_track", lambda **kw: (120.0, None))
    return 0.125


def accidentals_per_100_notes(xml: str) -> float:
    return 100.0 * xml.count("<accidental") / max(1, xml.count("<note"))


def key_fifths(xml: str):
    found = re.findall(r"<fifths>(-?\d+)</fifths>", xml)
    return int(found[0]) if found else None


class TestRepeatedNotesSurvive:
    """Four repeated eighth notes used to come out as one whole note."""

    def test_repeated_notes_at_one_pitch_stay_separate(
        self, tone_wav, midi_factory, at_120bpm
    ):
        # Four eighth notes on middle C with a realistic 25ms re-articulation gap.
        midi = midi_factory([
            (60, 0.000, 0.225),
            (60, 0.250, 0.475),
            (60, 0.500, 0.725),
            (60, 0.750, 0.975),
        ])

        out, _ = pipeline.quantize_to_grid(midi, tone_wav)

        assert len(out.instruments[0].notes) == 4

    def test_the_repeated_notes_keep_their_rhythm(self, tone_wav, midi_factory, at_120bpm):
        midi = midi_factory([(60, t, t + 0.225) for t in (0.0, 0.25, 0.5, 0.75)])

        out, _ = pipeline.quantize_to_grid(midi, tone_wav)

        starts = [round(n.start, 3) for n in out.instruments[0].notes]
        assert starts == [0.0, 0.25, 0.5, 0.75]

    def test_a_sustained_note_split_by_the_model_is_still_rejoined(
        self, tone_wav, midi_factory, at_120bpm
    ):
        """The behaviour merging exists for must not regress."""
        midi = midi_factory([
            (60, 0.000, 0.260),
            (60, 0.262, 0.520),   # 2ms apart: a model artefact, not a re-attack
            (60, 0.521, 0.780),
        ])

        out, _ = pipeline.quantize_to_grid(midi, tone_wav)

        notes = out.instruments[0].notes
        assert len(notes) == 1
        assert notes[0].end - notes[0].start == pytest.approx(0.75, abs=0.13)

    def test_repeated_drum_hits_are_untouched(self, tone_wav, midi_factory, at_120bpm):
        midi = midi_factory([(42, t, t + 0.05) for t in (0.0, 0.125, 0.25, 0.375)],
                            is_drum=True)

        out, _ = pipeline.quantize_to_grid(midi, tone_wav)

        assert len(out.instruments[0].notes) == 4

    @pytest.mark.parametrize("gap", [0.021, 0.05, 0.12])
    def test_gaps_above_the_tolerance_are_real_re_articulations(self, gap):
        notes = [
            pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.2),
            pretty_midi.Note(velocity=90, pitch=60, start=0.2 + gap, end=0.4),
        ]

        merged = pipeline._merge_same_pitch(notes, gap_tolerance=pipeline.MERGE_GAP_SECONDS)

        assert len(merged) == 2

    @pytest.mark.parametrize("gap", [0.0, 0.005, 0.019])
    def test_gaps_below_the_tolerance_are_model_artefacts(self, gap):
        notes = [
            pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.2),
            pretty_midi.Note(velocity=90, pitch=60, start=0.2 + gap, end=0.4),
        ]

        merged = pipeline._merge_same_pitch(notes, gap_tolerance=pipeline.MERGE_GAP_SECONDS)

        assert len(merged) == 1


class TestMonophonicPartsStaySingleLine:
    """Independent rounding could put two pitches in one grid step."""

    def test_a_short_note_stretched_onto_its_neighbour_does_not_make_a_chord(
        self, tone_wav, midi_factory, at_120bpm
    ):
        # The first note is shorter than a grid step, so it is stretched to a
        # full step and lands exactly on top of the second.
        midi = midi_factory([(40, 0.10, 0.14), (43, 0.14, 0.30)])

        out, _ = pipeline.quantize_to_grid(midi, tone_wav, monophonic=True)

        notes = sorted(out.instruments[0].notes, key=lambda n: n.start)
        overlaps = [(a.pitch, b.pitch) for a, b in zip(notes, notes[1:]) if b.start < a.end]
        assert overlaps == []

    def test_the_polyphonic_path_is_unaffected(self, tone_wav, midi_factory, at_120bpm):
        """A piano chord is three simultaneous notes and must stay that way."""
        midi = midi_factory([(60, 0.0, 0.5), (64, 0.0, 0.5), (67, 0.0, 0.5)])

        out, _ = pipeline.quantize_to_grid(midi, tone_wav, monophonic=False)

        assert len(out.instruments[0].notes) == 3

    def test_an_overlapping_note_is_cut_at_the_next_onset(self):
        notes = [
            pretty_midi.Note(velocity=90, pitch=40, start=0.0, end=0.50),
            pretty_midi.Note(velocity=90, pitch=45, start=0.25, end=0.75),
        ]

        mono = pipeline._enforce_monophony(notes)

        assert len(mono) == 2
        assert mono[0].end == pytest.approx(0.25)
        assert mono[1].start == pytest.approx(0.25)

    def test_notes_sharing_an_onset_resolve_to_the_longer_one(self):
        notes = [
            pretty_midi.Note(velocity=90, pitch=40, start=0.0, end=0.20),
            pretty_midi.Note(velocity=90, pitch=45, start=0.0, end=0.50),
        ]

        mono = pipeline._enforce_monophony(notes)

        assert len(mono) == 1
        assert mono[0].pitch == 45

    def test_a_line_with_no_collisions_is_left_alone(self):
        notes = [
            pretty_midi.Note(velocity=90, pitch=40, start=0.0, end=0.25),
            pretty_midi.Note(velocity=90, pitch=45, start=0.25, end=0.50),
        ]

        mono = pipeline._enforce_monophony(notes)

        assert [(n.pitch, n.start, n.end) for n in mono] == \
               [(40, 0.0, 0.25), (45, 0.25, 0.50)]

    def test_no_note_is_left_with_zero_length(self):
        notes = [
            pretty_midi.Note(velocity=90, pitch=40, start=0.0, end=0.5),
            pretty_midi.Note(velocity=90, pitch=45, start=0.1, end=0.6),
            pretty_midi.Note(velocity=90, pitch=47, start=0.2, end=0.7),
        ]

        for n in pipeline._enforce_monophony(notes):
            assert n.end > n.start

    def test_empty_input_is_handled(self):
        assert pipeline._enforce_monophony([]) == []

    def test_bass_and_vocals_are_the_monophonic_stems(self):
        assert transcribers.MONOPHONIC_STEMS == {"bass", "vocals"}
        assert transcribers.MONOPHONIC_STEMS <= set(pipeline.VALID_STEMS)


class TestKeySignature:
    """Every score used to be engraved in C major, whatever the piece."""

    def _score_in(self, tmp_path, pitches, stem=None):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(program=0)
        inst.notes = [
            pretty_midi.Note(velocity=90, pitch=p, start=i * 0.5, end=i * 0.5 + 0.5)
            for i, p in enumerate(pitches)
        ]
        midi.instruments.append(inst)
        return pipeline.notate(midi, str(tmp_path / "score.musicxml"), stem=stem)

    def test_a_key_signature_is_written(self, tmp_path):
        # E-flat major scale, twice, so the analysis has something to work with.
        e_flat = [63, 65, 67, 68, 70, 72, 74, 75] * 2

        xml = self._score_in(tmp_path, e_flat)

        assert "<fifths>" in xml

    def test_a_flat_key_is_detected_as_flats(self, tmp_path):
        e_flat = [63, 65, 67, 68, 70, 72, 74, 75] * 2

        assert key_fifths(self._score_in(tmp_path, e_flat)) == -3

    def test_a_sharp_key_is_detected_as_sharps(self, tmp_path):
        d_major = [62, 64, 66, 67, 69, 71, 73, 74] * 2

        assert key_fifths(self._score_in(tmp_path, d_major)) == 2

    def test_c_major_gets_an_empty_key_signature(self, tmp_path):
        c_major = [60, 62, 64, 65, 67, 69, 71, 72] * 2

        assert key_fifths(self._score_in(tmp_path, c_major)) == 0

    def test_the_key_signature_removes_most_accidentals(self, tmp_path):
        """The whole point: notes covered by the key stop printing accidentals."""
        e_flat = [63, 65, 67, 68, 70, 72, 74, 75] * 3

        with_key = self._score_in(tmp_path, e_flat)

        assert accidentals_per_100_notes(with_key) < 10

    def test_pitches_are_respelled_toward_the_key(self, tmp_path):
        """In E-flat major a black key is B-flat, not A-sharp."""
        e_flat = [63, 65, 67, 68, 70, 72, 74, 75] * 2

        xml = self._score_in(tmp_path, e_flat)

        assert "<step>A</step>\n          <alter>1</alter>" not in xml
        assert xml.count("<alter>-1</alter>") >= xml.count("<alter>1</alter>")

    def test_drums_are_not_given_a_key(self, tmp_path):
        """Percussion has no key and its notes are unpitched."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        drum = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
        drum.notes = [
            pretty_midi.Note(velocity=100, pitch=36 if i % 2 else 42,
                             start=i * 0.25, end=i * 0.25 + 0.1)
            for i in range(8)
        ]
        midi.instruments.append(drum)

        xml = pipeline.notate(midi, str(tmp_path / "drums.musicxml"), stem="drums")

        assert "score-partwise" in xml

    def test_a_score_too_small_to_analyse_still_engraves(self, tmp_path):
        xml = self._score_in(tmp_path, [60])
        assert "score-partwise" in xml

    def test_the_piano_grand_staff_gets_the_key_on_both_staves(self, tmp_path):
        e_flat = [51, 56, 63, 65, 67, 68, 70, 72, 74, 75] * 2

        xml = self._score_in(tmp_path, e_flat, stem="piano")

        assert xml.count("<fifths>") == 2

    def test_the_notes_themselves_are_unchanged(self, tmp_path):
        """Respelling is enharmonic: it must not move any pitch."""
        e_flat = [63, 65, 67, 68, 70, 72, 74, 75] * 2
        xml = self._score_in(tmp_path, e_flat)

        path = tmp_path / "score.musicxml"
        path.write_text(xml, encoding="utf-8")
        sounding = [p.midi for n in converter.parse(str(path)).recurse().notes
                    for p in n.pitches]

        assert sorted(sounding) == sorted(e_flat)


class TestSustainedNotesSurviveVibrato:
    """pYIN's rounded pitch flickers under vibrato; the note was discarded."""

    def test_a_note_with_wide_vibrato_is_notated(self, tone_factory):
        path = tone_factory(midi_pitch=57.0, duration=3.0, vibrato_semitones=1.0)

        pm = transcribers._f0_to_midi(path, fmin=82.0, fmax=1000.0, program=53)

        notated = sum(n.end - n.start for n in pm.instruments[0].notes)
        assert notated > 0.8 * 3.0, f"only {notated:.2f}s of 3.0s survived"

    def test_it_comes_out_as_one_note_not_a_shredded_line(self, tone_factory):
        path = tone_factory(midi_pitch=57.0, duration=3.0, vibrato_semitones=1.0)

        pm = transcribers._f0_to_midi(path, fmin=82.0, fmax=1000.0, program=53)

        assert len(pm.instruments[0].notes) <= 2

    def test_the_centre_pitch_is_reported(self, tone_factory):
        path = tone_factory(midi_pitch=57.0, duration=3.0, vibrato_semitones=1.0)

        pm = transcribers._f0_to_midi(path, fmin=82.0, fmax=1000.0, program=53)

        notes = pm.instruments[0].notes
        longest = max(notes, key=lambda n: n.end - n.start)
        assert abs(longest.pitch - 57) <= 1

    def test_a_fast_scale_is_not_smoothed_away(self, tmp_path):
        """Smoothing must not cost resolution on real melodic movement."""
        import numpy as np
        import soundfile as sf

        sr = 22050
        scale = [60, 62, 64, 65, 67, 69, 71, 72]  # 16th notes at 120 BPM
        chunks = []
        for m in scale:
            t = np.linspace(0, 0.125, int(sr * 0.125), endpoint=False)
            phase = 2 * np.pi * np.cumsum(librosa.midi_to_hz(np.full_like(t, m))) / sr
            y = 0.6 * np.sin(phase) + 0.2 * np.sin(2 * phase)
            fade = int(0.005 * sr)
            y[:fade] *= np.linspace(0, 1, fade)
            y[-fade:] *= np.linspace(1, 0, fade)
            chunks.append(y)
        path = tmp_path / "scale.wav"
        sf.write(str(path), np.concatenate(chunks).astype("float32"), sr)

        pm = transcribers._f0_to_midi(str(path), fmin=82.0, fmax=1000.0, program=53)

        found = {n.pitch for n in pm.instruments[0].notes}
        assert len(found & set(scale)) >= 6, f"only recovered {sorted(found)}"


class TestPitchSmoothing:
    def test_an_isolated_outlier_is_removed(self):
        import numpy as np

        track = np.array([57.0] * 4 + [64.0] + [57.0] * 4)

        smoothed = transcribers._smooth_pitch_track(track, 5)

        assert smoothed[4] == pytest.approx(57.0)

    def test_unvoiced_frames_stay_unvoiced(self):
        import numpy as np

        track = np.array([57.0, 57.0, np.nan, np.nan, 57.0])

        smoothed = transcribers._smooth_pitch_track(track, 5)

        assert np.isnan(smoothed[2]) and np.isnan(smoothed[3])

    def test_a_kernel_of_one_is_a_no_op(self):
        import numpy as np

        track = np.array([57.0, 64.0, 57.0])

        assert np.array_equal(transcribers._smooth_pitch_track(track, 1), track)

    def test_an_all_unvoiced_track_is_handled(self):
        import numpy as np

        track = np.full(10, np.nan)

        assert np.isnan(transcribers._smooth_pitch_track(track, 9)).all()

    def test_a_step_between_two_pitches_is_preserved(self):
        """A median filter must not smear a real note change into a glide."""
        import numpy as np

        track = np.array([57.0] * 10 + [60.0] * 10)

        smoothed = transcribers._smooth_pitch_track(track, 9)

        assert set(np.unique(smoothed)) == {57.0, 60.0}
