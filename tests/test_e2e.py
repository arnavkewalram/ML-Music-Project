"""End-to-end test of the real chain: Demucs -> transcriber -> music21.

Marked `slow` and excluded from the default run: it loads several hundred
megabytes of pretrained models. Run it explicitly with `pytest -m slow`.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import pytest

from webapp import pipeline, server

pytestmark = pytest.mark.slow

# The shortest bundled sample, so the test stays under a minute once models are
# warm. It is a solo trumpet, which Demucs places in the "other" stem.
SAMPLE = os.path.join(server.SAMPLES_DIR, "trumpet.ogg")


@pytest.fixture(scope="module")
def transcription(tmp_path_factory):
    if not os.path.isfile(SAMPLE):
        pytest.skip(f"sample audio missing: {SAMPLE}")
    work_dir = tmp_path_factory.mktemp("e2e")
    return pipeline.run(SAMPLE, "other", str(work_dir))


def test_the_chain_produces_all_three_artefacts(transcription):
    for key in ("stem_path", "midi_path", "musicxml_path"):
        assert os.path.isfile(transcription[key]), key


def test_the_isolated_stem_is_readable_audio(transcription):
    import librosa

    y, sr = librosa.load(transcription["stem_path"], sr=None)
    assert len(y) > 0
    assert sr > 0


def test_the_midi_is_readable_and_non_empty(transcription):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(transcription["midi_path"])
    notes = [n for inst in pm.instruments for n in inst.notes]
    assert notes, "a solo trumpet line should yield notes"
    assert len(notes) == transcription["n_notes"]


def test_the_musicxml_parses_as_a_score(transcription):
    from music21 import converter

    score = converter.parse(transcription["musicxml_path"])
    assert list(score.recurse().notes)


def test_reported_duration_is_the_input_duration(transcription):
    import librosa

    assert transcription["duration"] == pytest.approx(
        librosa.get_duration(path=SAMPLE), abs=0.5
    )


def test_the_reported_tempo_is_musically_plausible(transcription):
    assert 30 <= transcription["tempo"] <= 300


def test_every_note_lands_on_the_quantization_grid(transcription):
    """Onsets sit on 16th notes, which is what makes the rhythm readable.

    Tolerance is 2ms rather than exact: the MIDI round-trip stores times as
    ticks, and the reported tempo is rounded to one decimal while the grid uses
    the full-precision estimate. Both drifts are far below anything notatable.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(transcription["midi_path"])
    grid = (60.0 / transcription["tempo"]) / 4

    for note in (n for inst in pm.instruments for n in inst.notes):
        remainder = note.start % grid
        off_grid_by = min(remainder, grid - remainder)
        assert off_grid_by < 2e-3, f"note at {note.start}s is {off_grid_by}s off the grid"


def test_tempo_is_a_property_of_the_song_not_of_the_stem(tmp_path):
    """Two instruments pulled from one song must agree on its tempo.

    Regression: beat-tracking an isolated stem (an offbeat skank, a sparse bass)
    gives half- or double-time answers, so the two parts could not be played
    together. Tempo is now taken from the full mix.
    """
    if not os.path.isfile(SAMPLE):
        pytest.skip("sample audio missing")

    a = pipeline.run(SAMPLE, "other", str(tmp_path / "a"))
    b = pipeline.run(SAMPLE, "vocals", str(tmp_path / "b"))

    assert a["tempo"] == b["tempo"]
