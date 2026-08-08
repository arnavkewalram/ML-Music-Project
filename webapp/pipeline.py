"""
Transcription pipeline used by the web server.

    song -> Demucs (separate one instrument)
         -> per-instrument specialist transcriber (see transcribers.py)
         -> music21 (light quantization + MusicXML)

Demucs (separation) is loaded once and cached here; the per-instrument
transcription models are owned by transcribers.py.
"""

import os
import shutil
import threading
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import librosa
import soundfile as sf

from webapp import transcribers

# Demucs 6-stem source order: drums, bass, other, vocals, guitar, piano
DEMUCS_MODEL = "htdemucs_6s"
VALID_STEMS = ["drums", "bass", "other", "vocals", "guitar", "piano"]

# Demucs holds the whole waveform in memory and runs in roughly real time on
# CPU, so an unbounded input is both an OOM risk and a request that never
# returns. Six minutes covers all but the longest songs; anything past that is
# transcribed up to the cap and the caller is told it was truncated.
MAX_ANALYSIS_SECONDS = 360.0

# Serialises the heavy stages. Two concurrent Demucs runs need twice the peak
# RAM for no throughput gain on one machine, and queueing is a better failure
# mode than the OOM killer. FastAPI runs sync endpoints in a threadpool, so
# waiting here just queues the request.
_INFERENCE_LOCK = threading.Lock()

_demucs = None
_demucs_device = None
_demucs_lock = threading.Lock()


def _get_demucs():
    """Load Demucs once. Double-checked locking: the threadpool can race here."""
    global _demucs, _demucs_device
    if _demucs is None:
        with _demucs_lock:
            if _demucs is None:
                import torch
                from demucs.pretrained import get_model
                model = get_model(DEMUCS_MODEL)
                model.eval()
                # Prefer MPS (Apple GPU) when available, fall back to CPU
                _demucs_device = "mps" if torch.backends.mps.is_available() else "cpu"
                _demucs = model  # publish last: readers see a fully-built model
    return _demucs, _demucs_device


def analysis_window(audio_path: str):
    """Return (full_duration, analysed_duration, was_truncated) for an input."""
    full = float(librosa.get_duration(path=audio_path))
    used = min(full, MAX_ANALYSIS_SECONDS)
    return full, used, full > MAX_ANALYSIS_SECONDS + 1e-3


def prune_runs(runs_dir: str, keep: int) -> int:
    """Delete all but the `keep` most recent run directories; return how many went.

    Each run leaves a stem WAV, a MIDI and a MusicXML on disk — tens of
    megabytes. Without this the directory grows without bound for as long as the
    server is used.
    """
    if not os.path.isdir(runs_dir):
        return 0
    entries = [
        (os.path.getmtime(p), p)
        for p in (os.path.join(runs_dir, n) for n in os.listdir(runs_dir))
        if os.path.isdir(p)
    ]
    entries.sort(reverse=True)
    removed = 0
    for _, path in entries[keep:]:
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


def warmup():
    """Load separation + transcription models ahead of the first request."""
    _get_demucs()
    transcribers.warmup()


def separate(audio_path: str, stem: str, out_path: str,
             max_seconds: float = MAX_ANALYSIS_SECONDS) -> str:
    """Isolate one instrument stem from a full mix and write it to out_path."""
    import torch
    from demucs.apply import apply_model

    if stem not in VALID_STEMS:
        raise ValueError(f"stem must be one of {VALID_STEMS}, got '{stem}'")

    model, device = _get_demucs()
    sr = model.samplerate

    audio, _ = librosa.load(audio_path, sr=sr, mono=False, duration=max_seconds)
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


def quantize_to_grid(midi_data, audio_path: str, subdivision: int = 4,
                     max_seconds: float = MAX_ANALYSIS_SECONDS,
                     monophonic: bool = False):
    """Estimate the real tempo and snap every note onset/offset to a beat grid.

    The transcription models emit notes at arbitrary millisecond times, so when
    music21 notates them against a default 120 BPM the durations land on ragged
    tuplets (we measured 10-21 distinct rhythmic values per score). Snapping to a
    16th-note grid at the *actual* tempo collapses that to a handful of clean
    durations. Returns (new_pretty_midi, tempo_bpm).
    """
    import pretty_midi

    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=max_seconds)
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
        notes = inst.notes
        # Drums are discrete hits: never merged, never made monophonic.
        if not inst.is_drum:
            # Rejoin the slivers of one sustained note BEFORE snapping, while
            # the real gaps are still visible.
            notes = _merge_same_pitch(notes, gap_tolerance=MERGE_GAP_SECONDS)

        snapped = []
        for n in notes:
            start = round(n.start / grid) * grid
            end = round(n.end / grid) * grid
            if end <= start:
                end = start + grid  # keep at least one grid step
            snapped.append(pretty_midi.Note(
                velocity=n.velocity, pitch=n.pitch, start=start, end=end))

        if not inst.is_drum:
            # Only genuine overlaps are left to merge. Notes that merely touch
            # are repeated notes and must stay separate.
            snapped = _merge_same_pitch(snapped, gap_tolerance=-1e-6)
        if monophonic and not inst.is_drum:
            snapped = _enforce_monophony(snapped)

        ni.notes = snapped
        out.instruments.append(ni)
    return out, round(tempo, 1)


# Same-pitch fragments closer together than this were one sustained note that
# the model split, not two notes the player articulated. Re-attacking a note
# takes longer than 20ms; model slivers are separated by essentially nothing.
#
# This has to be applied before quantization: once onsets are snapped to a 16th
# grid, a 25ms re-articulation gap and a 0ms artefact both become exactly 0, and
# four repeated eighth notes merge into a single whole note.
MERGE_GAP_SECONDS = 0.02


def _merge_same_pitch(notes, gap_tolerance: float = 0.0):
    """Merge same-pitch notes separated by at most `gap_tolerance` seconds.

    Transcription models frequently split a single sustained note into several
    slivers; collapsing them removes spurious rhythmic values and fake overlaps
    without changing what is actually played. A negative tolerance merges only
    notes that genuinely overlap, leaving touching notes alone.
    """
    import pretty_midi

    by_pitch = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    merged = []
    for pitch, group in by_pitch.items():
        group = sorted(group, key=lambda x: x.start)
        cur = None
        for n in group:
            if cur is not None and n.start <= cur.end + gap_tolerance:
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


def _enforce_monophony(notes):
    """Reduce a part to one sounding note at a time.

    A bass or a voice cannot play a chord, but snapping each note's start and
    end independently can leave two different pitches occupying the same grid
    step — which engraves as a two-note chord in a single-line part. Where notes
    collide, the earlier one is cut short at the next onset; where they share an
    onset, the longer one wins.
    """
    import pretty_midi

    ordered = sorted(notes, key=lambda n: (n.start, -(n.end - n.start), n.pitch))
    kept = []
    for n in ordered:
        if kept and n.start < kept[-1].end - 1e-9:
            if n.start <= kept[-1].start + 1e-9:
                continue  # same onset — the longer note already took the slot
            prev = kept[-1]
            kept[-1] = pretty_midi.Note(velocity=prev.velocity, pitch=prev.pitch,
                                        start=prev.start, end=n.start)
        kept.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                     start=n.start, end=n.end))
    return kept


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


def _apply_key_signature(score):
    """Detect the key, respell accidentals to suit it, and engrave it.

    MIDI carries no key, so music21 notates everything in C major and spells
    every black key as a sharp. Chopin's Grande Valse — E-flat major — came out
    with 46 accidentals per 100 notes, which is unreadable.

    Three steps, all of which are needed:
      1. Krumhansl-Schmuckler key analysis (music21's `analyze('key')`).
      2. Respell enharmonically toward the key's accidental direction, so a
         flat-key piece stops spelling B-flat as A-sharp.
      3. Recompute accidental display per measure against the new key
         signature, so notes already covered by it stop printing accidentals.

    Measured on the same piece: 46.4 -> 9.3 accidentals per 100 notes.
    Returns the detected key, or None if analysis was not possible.
    """
    from music21 import key as m21key, stream

    detected = score.analyze("key")
    signature = m21key.KeySignature(detected.sharps)
    prefer_flats = detected.sharps < 0

    for element in score.recurse().notes:
        for p in getattr(element, "pitches", ()):
            if p.accidental is None:
                continue
            wrong_direction = (p.accidental.alter > 0) if prefer_flats else (p.accidental.alter < 0)
            if wrong_direction:
                p.getEnharmonic(inPlace=True)
            if p.accidental is not None:
                # Clear the spelling decision inherited from MIDI so
                # makeAccidentals can decide against the key signature.
                p.accidental.displayStatus = None

    for part in (score.parts if len(score.parts) else [score]):
        part.insert(0, signature)
        for measure in part.getElementsByClass(stream.Measure):
            measure.makeAccidentals(useKeySignature=signature, inPlace=True,
                                    overrideStatus=True)
    return detected


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
        # Percussion has no key, and its notes are unpitched, so key analysis
        # neither means anything nor works there.
        if stem != "drums":
            try:
                _apply_key_signature(score)
            except Exception:
                pass  # an unkeyed score is still a usable score
        score.write("musicxml", fp=out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        os.unlink(mid_tmp)


def _stem_energy_ratio(mix_path: str, stem_path: str,
                       max_seconds: float = MAX_ANALYSIS_SECONDS) -> float:
    """RMS energy of the isolated stem relative to the full mix.

    If an instrument isn't really in the track, Demucs returns a near-silent
    stem and the transcriber still hallucinates notes from bleed. A low ratio
    flags 'this instrument probably isn't here'.

    Both sides are read over the same window: the stem only covers the analysed
    portion, so comparing it against an untruncated mix would understate the
    ratio and wrongly report the instrument as absent.
    """
    try:
        mix, _ = librosa.load(mix_path, sr=22050, mono=True, duration=max_seconds)
        stem, _ = librosa.load(stem_path, sr=22050, mono=True, duration=max_seconds)
        mix_rms = float(np.sqrt(np.mean(mix ** 2)))
        stem_rms = float(np.sqrt(np.mean(stem ** 2)))
        return stem_rms / (mix_rms + 1e-8)
    except Exception:
        return 1.0  # on error, don't flag


def run(audio_path: str, stem: str, work_dir: str) -> dict:
    """Full chain. Returns paths + the MusicXML string + simple stats."""
    os.makedirs(work_dir, exist_ok=True)
    stem_path = os.path.join(work_dir, f"stem_{stem}.wav")
    midi_path = os.path.join(work_dir, "transcription.mid")
    xml_path = os.path.join(work_dir, "sheet_music.musicxml")

    # Report the INPUT audio duration, not the MIDI end time (which is 0 when an
    # absent instrument yields no notes — confusing "duration: 0.0").
    try:
        duration, analysed, truncated = analysis_window(audio_path)
    except Exception:
        duration, analysed, truncated = 0.0, MAX_ANALYSIS_SECONDS, False

    # Everything from here on loads a full waveform into memory and runs a
    # neural net over it; one at a time.
    with _INFERENCE_LOCK:
        separate(audio_path, stem, stem_path, max_seconds=analysed)
        stem_ratio = _stem_energy_ratio(audio_path, stem_path, max_seconds=analysed)
        midi_data, method = transcribers.transcribe(stem, stem_path)
        # Estimate tempo from the FULL MIX, not the isolated stem: a stem (e.g. an
        # offbeat drum skank) fools beat-tracking into half/double tempo, whereas the
        # mix gives one robust tempo shared by every instrument in the song.
        midi_data, tempo = quantize_to_grid(
            midi_data, audio_path, max_seconds=analysed,
            monophonic=stem in transcribers.MONOPHONIC_STEMS)

    midi_data.write(midi_path)
    musicxml = notate(midi_data, xml_path, stem=stem)

    if not duration:
        duration = float(midi_data.get_end_time())

    n_notes = sum(len(inst.notes) for inst in midi_data.instruments)
    return {
        "stem_path": stem_path,
        "midi_path": midi_path,
        "musicxml_path": xml_path,
        "musicxml": musicxml,
        "n_notes": n_notes,
        "duration": duration,
        "analyzed_duration": analysed,
        "truncated": truncated,
        "method": method,
        "tempo": tempo,
        "stem_energy_ratio": round(stem_ratio, 4),
    }
