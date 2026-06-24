"""
Transcription pipeline used by the web server.

    song -> Demucs (separate one instrument)
         -> per-instrument specialist transcriber (see transcribers.py)
         -> music21 (light quantization + MusicXML)

Demucs (separation) is loaded once and cached here; the per-instrument
transcription models are owned by transcribers.py.
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import librosa
import soundfile as sf

from webapp import transcribers

# Demucs 6-stem source order: drums, bass, other, vocals, guitar, piano
DEMUCS_MODEL = "htdemucs_6s"
VALID_STEMS = ["drums", "bass", "other", "vocals", "guitar", "piano"]

_demucs = None
_demucs_device = None


def _get_demucs():
    global _demucs, _demucs_device
    if _demucs is None:
        import torch
        from demucs.pretrained import get_model
        _demucs = get_model(DEMUCS_MODEL)
        _demucs.eval()
        # Prefer MPS (Apple GPU) when available, fall back to CPU
        _demucs_device = "mps" if torch.backends.mps.is_available() else "cpu"
    return _demucs, _demucs_device


def warmup():
    """Load separation + transcription models ahead of the first request."""
    _get_demucs()
    transcribers.warmup()


def separate(audio_path: str, stem: str, out_path: str) -> str:
    """Isolate one instrument stem from a full mix and write it to out_path."""
    import torch
    from demucs.apply import apply_model

    if stem not in VALID_STEMS:
        raise ValueError(f"stem must be one of {VALID_STEMS}, got '{stem}'")

    model, device = _get_demucs()
    sr = model.samplerate

    audio, _ = librosa.load(audio_path, sr=sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    elif audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)

    wav = torch.from_numpy(audio).float()
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    try:
        with torch.no_grad():
            sources = apply_model(model, wav[None], device=device, progress=False)[0]
    except Exception:
        # Some ops can be unsupported on MPS; retry on CPU
        with torch.no_grad():
            sources = apply_model(model, wav[None], device="cpu", progress=False)[0]

    sources = sources * ref.std() + ref.mean()
    idx = model.sources.index(stem)
    sf.write(out_path, sources[idx].cpu().numpy().T, sr)
    return out_path


def quantize_to_grid(midi_data, audio_path: str, subdivision: int = 4):
    """Estimate the real tempo and snap every note onset/offset to a beat grid.

    The transcription models emit notes at arbitrary millisecond times, so when
    music21 notates them against a default 120 BPM the durations land on ragged
    tuplets (we measured 10-21 distinct rhythmic values per score). Snapping to a
    16th-note grid at the *actual* tempo collapses that to a handful of clean
    durations. Returns (new_pretty_midi, tempo_bpm).
    """
    import pretty_midi

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
    except Exception:
        tempo = 0.0
    # librosa's estimate jitters; a real 40 BPM track can read 39.8, so don't
    # discard the 30-45 range. Only fall back for absurd/failed estimates.
    if not tempo or tempo < 30 or tempo > 300:
        tempo = 120.0

    grid = (60.0 / tempo) / subdivision  # seconds per grid step (16th note)
    out = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for inst in midi_data.instruments:
        ni = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        snapped = []
        for n in inst.notes:
            start = round(n.start / grid) * grid
            end = round(n.end / grid) * grid
            if end <= start:
                end = start + grid  # keep at least one grid step
            snapped.append(pretty_midi.Note(
                velocity=n.velocity, pitch=n.pitch, start=start, end=end))
        # Drums are discrete hits; melodic notes get fragments merged.
        ni.notes = snapped if inst.is_drum else _merge_same_pitch(snapped)
        out.instruments.append(ni)
    return out, round(tempo, 1)


def _merge_same_pitch(notes):
    """Merge contiguous/overlapping same-pitch notes into one and drop duplicates.

    Transcription models frequently split a single sustained note into several
    grid-length slivers; collapsing them removes spurious rhythmic values and
    fake overlaps without changing what is actually played.
    """
    import pretty_midi

    by_pitch = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    merged = []
    for pitch, group in by_pitch.items():
        group.sort(key=lambda x: x.start)
        cur = None
        for n in group:
            if cur is not None and n.start <= cur.end + 1e-6:  # contiguous/overlapping
                cur.end = max(cur.end, n.end)
                cur.velocity = max(cur.velocity, n.velocity)
            else:
                if cur is not None:
                    merged.append(cur)
                cur = pretty_midi.Note(velocity=n.velocity, pitch=pitch,
                                       start=n.start, end=n.end)
        if cur is not None:
            merged.append(cur)
    merged.sort(key=lambda x: (x.start, x.pitch))
    return merged


GRAND_STAFF_SPLIT = 60  # middle C: notes >= go on treble, below on bass


def _to_grand_staff(score, split: int = GRAND_STAFF_SPLIT):
    """Split a single-part score into a piano grand staff (treble + bass clef).

    Real piano notation always uses two braced staves; cramming everything onto
    one staff is both wrong and unreadable for dense pieces. Distribute each
    note/chord by pitch across two staves and brace them together.
    """
    from music21 import stream, clef, note, chord, layout

    treble = stream.Part()
    treble.insert(0, clef.TrebleClef())
    bass = stream.Part()
    bass.insert(0, clef.BassClef())

    # Flatten so each note carries its ABSOLUTE offset (recurse() would give the
    # within-measure offset and collapse everything onto the first beats).
    for el in score.flatten().notes:
        ql = el.duration.quarterLength
        off = el.offset
        if isinstance(el, chord.Chord):
            hi = [p for p in el.pitches if p.midi >= split]
            lo = [p for p in el.pitches if p.midi < split]
            if hi:
                treble.insert(off, chord.Chord(hi, quarterLength=ql))
            if lo:
                bass.insert(off, chord.Chord(lo, quarterLength=ql))
        else:
            dest = treble if el.pitch.midi >= split else bass
            dest.insert(off, note.Note(el.pitch, quarterLength=ql))

    treble.makeNotation(inPlace=True)
    bass.makeNotation(inPlace=True)

    grand = stream.Score()
    grand.insert(0, treble)
    grand.insert(0, bass)
    grand.insert(0, layout.StaffGroup([treble, bass], symbol="brace"))
    return grand


def notate(midi_data, out_path: str, stem: str = None, quantize: bool = True) -> str:
    """Convert a pretty_midi transcription to MusicXML; return the XML as a string."""
    import tempfile
    from music21 import converter as m21converter

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        mid_tmp = tmp.name
        midi_data.write(mid_tmp)
    try:
        score = m21converter.parse(mid_tmp)
        if quantize:
            try:
                # Snap to 16th-note and triplet grid so the rhythm is readable
                score.quantize((4, 3), inPlace=True, recurse=True)
            except Exception:
                pass
        if stem == "piano":
            try:
                score = _to_grand_staff(score)
            except Exception:
                pass  # fall back to the single-staff score
        score.write("musicxml", fp=out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        os.unlink(mid_tmp)


def run(audio_path: str, stem: str, work_dir: str) -> dict:
    """Full chain. Returns paths + the MusicXML string + simple stats."""
    os.makedirs(work_dir, exist_ok=True)
    stem_path = os.path.join(work_dir, f"stem_{stem}.wav")
    midi_path = os.path.join(work_dir, "transcription.mid")
    xml_path = os.path.join(work_dir, "sheet_music.musicxml")

    separate(audio_path, stem, stem_path)
    midi_data, method = transcribers.transcribe(stem, stem_path)
    midi_data, tempo = quantize_to_grid(midi_data, stem_path)
    midi_data.write(midi_path)
    musicxml = notate(midi_data, xml_path, stem=stem)

    n_notes = sum(len(inst.notes) for inst in midi_data.instruments)
    # Report the INPUT audio duration, not the MIDI end time (which is 0 when an
    # absent instrument yields no notes — confusing "duration: 0.0").
    try:
        duration = float(librosa.get_duration(path=audio_path))
    except Exception:
        duration = float(midi_data.get_end_time())
    return {
        "stem_path": stem_path,
        "midi_path": midi_path,
        "musicxml_path": xml_path,
        "musicxml": musicxml,
        "n_notes": n_notes,
        "duration": duration,
        "method": method,
        "tempo": tempo,
    }
